#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
from itertools import combinations_with_replacement
import numpy as np
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ase.io
from ase.units import Hartree, Bohr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_cluster import radius
from torch_scatter import scatter_add
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF

from potentials import get_potential
from potentials.many_body_base import ManyBodyPotential

CONFIG_FILE = 'run.yaml'

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr

CONFIG = {}


# ── Unit conversion helpers ──────────────────────────────────────────

# Param kind -> unit conversion is defined per-potential via param_kinds().
# This avoids name collisions (e.g. 'A' means different units in Buck vs Yukawa).

def _user_to_internal(value, kind):
    """Convert a single parameter value from user units to internal."""
    if kind == 'energy':
        return value * EV_TO_HARTREE
    elif kind == 'distance':
        return value * ANGSTROM_TO_BOHR
    elif kind == 'inv_distance':
        return value * BOHR_TO_ANGSTROM  # 1/Ang -> 1/Bohr
    elif kind == 'energy_dist':
        return value * EV_TO_HARTREE * ANGSTROM_TO_BOHR
    elif kind == 'energy_dist6':
        return value * EV_TO_HARTREE * ANGSTROM_TO_BOHR ** 6
    elif kind == 'energy_dist8':
        return value * EV_TO_HARTREE * ANGSTROM_TO_BOHR ** 8
    elif kind == 'dimensionless':
        return value
    else:
        return value


def load_config(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath) as f:
        raw = yaml.safe_load(f)

    config = {}
    config.update(raw.get('training', {}))

    units = raw.get('units', {})
    config['input_energy_units'] = units.get('energy', 'ev')
    config['input_distance_units'] = units.get('distance', 'angstrom')

    config['datasets'] = raw.get('datasets', [])
    if not config['datasets']:
        raise ValueError("Config must specify at least one dataset under 'datasets'")

    config['initial_charges'] = {
        str(k): float(v) for k, v in (raw.get('charges') or {}).items()
    }
    config['formal_charges'] = {
        str(k): float(v) for k, v in (raw.get('formal_charges') or {}).items()
    }

    # ── Potential config ─────────────────────────────────────────
    pot_cfg = raw.get('potential', {})
    config['potential_type'] = pot_cfg.get('type', 'lj')
    config['coulomb'] = pot_cfg.get('coulomb', True)
    if 'cutoff_angstroms' in pot_cfg:
        config['cutoff_angstroms'] = pot_cfg['cutoff_angstroms']
    config['potential_defaults'] = pot_cfg.get('defaults', {})

    # ── Resolve potential class to get param names ───────────────
    pot_cls = get_potential(config['potential_type'])
    param_names = pot_cls.user_param_names()

    # ── Parse fixed_pairs and initial_guesses ────────────────────
    # Backward compatible: [val1, val2] list format works for LJ (epsilon, sigma)
    config['fixed_pairs'] = _parse_pair_params(
        raw.get('fixed_pairs') or {}, param_names
    )
    config['initial_guesses'] = _parse_pair_params(
        raw.get('initial_guesses') or {}, param_names
    )

    print(f"Loaded config: {filepath}")
    print(f"Potential: {config['potential_type']} "
          f"(params: {', '.join(param_names)})")
    return config


def _parse_pair_params(raw_pairs, param_names):
    """Parse pair parameter dicts from YAML.

    Supports two formats:
      - Dict format: {epsilon: 0.01, sigma: 2.5}
      - List format (backward compat for LJ): [0.01, 2.5]
    """
    parsed = {}
    for name, values in raw_pairs.items():
        if isinstance(values, dict):
            parsed[name] = {str(k): float(v) for k, v in values.items()}
        elif isinstance(values, (list, tuple)):
            if len(values) != len(param_names):
                raise ValueError(
                    f"Pair {name}: expected {len(param_names)} values "
                    f"({param_names}), got {len(values)}"
                )
            parsed[name] = {
                pn: float(v) for pn, v in zip(param_names, values)
            }
        else:
            raise ValueError(
                f"Pair {name}: expected dict or list, got {type(values)}"
            )
    return parsed


def compute_rdf_peaks(pair_names, xyz_files, box_size):
    """Compute RDF peak distances (in Bohr) for each pair type."""
    print("\nComputing equilibrium distances from RDF...")
    peak_distances = {}  # pair_name -> peak distance in Bohr
    box_dimensions = [box_size, box_size, box_size, 90.0, 90.0, 90.0]

    for xyz_path in xyz_files:
        if not os.path.exists(xyz_path):
            print(f"  File not found: {xyz_path}")
            continue

        try:
            universe = mda.Universe(xyz_path, topology_format='XYZ')
            num_frames = len(universe.trajectory)
            print(f"\n  Analyzing: {os.path.basename(xyz_path)} ({num_frames} frames)")
        except Exception as e:
            print(f"  Failed to load {xyz_path}: {e}")
            continue

        max_frames = min(100, num_frames)

        for pair_name in pair_names:
            if pair_name in peak_distances:
                continue

            element1, element2 = pair_name.split('-')
            selection1 = universe.select_atoms(f"name {element1}")
            selection2 = universe.select_atoms(f"name {element2}")
            if len(selection1) == 0 or len(selection2) == 0:
                continue

            try:
                for timestep in universe.trajectory[:max_frames]:
                    timestep.dimensions = box_dimensions
                universe.trajectory[0]

                rdf_analysis = InterRDF(
                    selection1, selection2,
                    nbins=200,
                    range=(0.5, min(15.0, box_size / 2)),
                )
                rdf_analysis.run(stop=max_frames)

                if np.max(rdf_analysis.results.rdf) < 1e-10:
                    continue

                smoothed_rdf = np.convolve(
                    rdf_analysis.results.rdf, np.ones(5) / 5, mode='same'
                )
                peak_distance_ang = rdf_analysis.results.bins[
                    np.argmax(smoothed_rdf)
                ]
                peak_distances[pair_name] = peak_distance_ang * ANGSTROM_TO_BOHR
                print(f"    {pair_name}: peak = {peak_distance_ang:.2f} A")
            except Exception as e:
                print(f"    {pair_name}: RDF failed ({e})")

    return peak_distances


class DFTDataset:
    def __init__(self, dataset_configs, cutoff_angstroms,
                 min_distance_angstroms, device):
        self.device = device
        self.cutoff_bohr = cutoff_angstroms * ANGSTROM_TO_BOHR
        self.min_distance_bohr = min_distance_angstroms * ANGSTROM_TO_BOHR
        self.frame_data = []
        self.forces_data = []
        self.graphs = []
        all_elements = set()

        print(f"\n{'='*60}")
        print("LOADING DATA")
        print('='*60)

        for dataset_index, dataset_config in enumerate(dataset_configs):
            dataset_name = dataset_config.get('name', f'Dataset {dataset_index}')
            print(f"\n{dataset_name}")

            frames = ase.io.read(
                dataset_config['xyz'], index=':', format='extxyz'
            )
            atom_symbols = frames[0].get_chemical_symbols()
            all_elements.update(atom_symbols)
            print(f"  Frames: {len(frames)}")

            if not frames:
                continue

            force_units = dataset_config.get('force_units', 'ev/ang')
            positions_bohr = []
            forces_au = []

            for frame in frames:
                positions_bohr.append(frame.positions * ANGSTROM_TO_BOHR)

                try:
                    frame_forces = frame.get_forces()
                    if force_units != 'au':
                        frame_forces = frame_forces / FORCE_AU_TO_EV_ANG
                    forces_au.append(frame_forces)
                except Exception:
                    forces_au.append(None)

            has_forces = any(f is not None for f in forces_au)
            print(f"  Forces: {'found' if has_forces else 'NOT FOUND'}")

            positions_bohr, forces_au = self._apply_force_filter(
                positions_bohr, forces_au, atom_symbols, dataset_config
            )

            self.frame_data.extend(
                [(atom_symbols, pos) for pos in positions_bohr]
            )
            self.forces_data.extend(forces_au)

        self.elements = sorted(all_elements)
        self.element_to_index = {
            element: index for index, element in enumerate(self.elements)
        }
        print(f"\nTotal: {len(self.frame_data)} frames, "
              f"Elements: {self.elements}")

    def _apply_force_filter(self, positions, forces, atom_symbols,
                            dataset_config):
        filter_percentile = (
            dataset_config.get('filter_percentile')
            or CONFIG.get('filter_percentile')
        )
        core_elements = set(CONFIG.get('core_elements', []))

        if not filter_percentile or not core_elements:
            return positions, forces

        core_atom_indices = [
            i for i, symbol in enumerate(atom_symbols)
            if symbol in core_elements
        ]
        if not core_atom_indices:
            return positions, forces

        max_force_per_frame = []
        for frame_forces in forces:
            if frame_forces is not None:
                core_forces = np.array(frame_forces)[core_atom_indices]
                max_force = np.max(np.linalg.norm(core_forces, axis=1))
                max_force_per_frame.append(max_force * FORCE_AU_TO_EV_ANG)
            else:
                max_force_per_frame.append(0.0)
        max_force_per_frame = np.array(max_force_per_frame)

        threshold = np.percentile(max_force_per_frame, filter_percentile)
        frames_to_keep = np.where(max_force_per_frame >= threshold)[0]

        print(f"  Filter: top {100 - filter_percentile:.0f}% by force "
              f"-> {len(frames_to_keep)} frames "
              f"(threshold={threshold:.3f} eV/A)")

        filtered_positions = [positions[i] for i in frames_to_keep]
        filtered_forces = [forces[i] for i in frames_to_keep]
        return filtered_positions, filtered_forces

    def build_graphs(self, pair_lookup_table, required_fields,
                     is_many_body=False):
        """Build PyG graphs with edge-level precomputed fields.

        Args:
            pair_lookup_table: (num_elements, num_elements) lookup tensor
            required_fields: set of field names the potential needs
                             (beyond inverse_distances and inverse_distances_sq
                             which are always computed for Coulomb)
            is_many_body: if True, precompute triplet indices for angular terms
        """
        print("Building graphs...", end=" ", flush=True)
        self.graphs = []

        need_distances = 'distances' in required_fields
        need_inv6 = 'inverse_distances_6' in required_fields
        need_inv12 = 'inverse_distances_12' in required_fields

        for i, (atom_symbols, positions) in enumerate(self.frame_data):
            pos_tensor = torch.tensor(
                positions, dtype=torch.float64, device=self.device
            )

            edge_index = radius(
                pos_tensor, pos_tensor,
                r=self.cutoff_bohr, max_num_neighbors=1024,
            )
            source = edge_index[0]
            target = edge_index[1]

            displacement = pos_tensor[target] - pos_tensor[source]
            distance = torch.norm(displacement, dim=1)

            valid_edges = (source != target) & (distance > self.min_distance_bohr)
            source = source[valid_edges]
            target = target[valid_edges]
            displacement = displacement[valid_edges]
            distance = distance[valid_edges]

            inverse_distance = 1.0 / distance
            unit_vectors = displacement * inverse_distance.unsqueeze(1)

            element_indices = torch.tensor(
                [self.element_to_index[s] for s in atom_symbols],
                dtype=torch.long, device=self.device,
            )

            frame_forces = self.forces_data[i]
            if frame_forces is not None:
                dft_forces = torch.tensor(
                    frame_forces, dtype=torch.float64, device=self.device
                )
            else:
                dft_forces = torch.zeros(
                    len(positions), 3, dtype=torch.float64, device=self.device
                )

            source_elements = element_indices[source]
            target_elements = element_indices[target]

            # Always-present fields
            kwargs = dict(
                pos=pos_tensor,
                edge_index=torch.stack([source, target]),
                inverse_distances=inverse_distance,
                inverse_distances_sq=inverse_distance ** 2,
                edge_unit_vectors=unit_vectors,
                element_indices=element_indices,
                source_elements=source_elements,
                target_elements=target_elements,
                pair_indices=pair_lookup_table[source_elements, target_elements],
                dft_forces=dft_forces,
            )

            # Conditional fields
            if need_distances:
                kwargs['distances'] = distance
            if need_inv6:
                kwargs['inverse_distances_6'] = inverse_distance ** 6
            if need_inv12:
                kwargs['inverse_distances_12'] = inverse_distance ** 12

            # Precompute triplet topology for many-body potentials
            if is_many_body:
                edge_idx = torch.stack([source, target])
                tri_i, tri_j, tri_k, eidx_ij, eidx_ik = \
                    ManyBodyPotential.build_triplets(
                        edge_idx, len(atom_symbols))
                kwargs['triplet_i'] = tri_i
                kwargs['triplet_j'] = tri_j
                kwargs['triplet_k'] = tri_k
                kwargs['edge_idx_ij'] = eidx_ij
                kwargs['edge_idx_ik'] = eidx_ik

            self.graphs.append(Data(**kwargs))

        print("Done.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class ForceFieldModel(nn.Module):
    """Generic force field model: pluggable potential + Coulomb."""

    def __init__(self, elements, pair_lookup_table, num_pairs,
                 potential, initial_charges, enforce_constraints=True,
                 coulomb=True):
        super().__init__()
        self.elements = elements
        self.num_elements = len(elements)
        self.num_pairs = num_pairs
        self.element_to_index = {e: i for i, e in enumerate(elements)}
        self.register_buffer('pair_lookup_table', pair_lookup_table)

        # Pluggable short-range potential (already initialized with params)
        self.potential = potential
        self.coulomb = coulomb
        self.is_many_body = isinstance(potential, ManyBodyPotential)

        # Charges (always present, even if coulomb=False, for neutrality penalty)
        self._setup_charge_constraints(
            elements, initial_charges, enforce_constraints
        )

    def _setup_charge_constraints(self, elements, initial_charges,
                                  enforce_constraints):
        self._cd_index = elements.index('Cd') if 'Cd' in elements else -1
        self._se_index = elements.index('Se') if 'Se' in elements else -1
        self._enforce_cd_se = (
            enforce_constraints
            and self._cd_index >= 0
            and self._se_index >= 0
        )

        if self._enforce_cd_se:
            independent_indices = [
                i for i in range(len(elements)) if i != self._se_index
            ]
        else:
            independent_indices = list(range(len(elements)))

        self.register_buffer(
            '_independent_charge_indices',
            torch.tensor(independent_indices, dtype=torch.long),
        )

        initial_charge_values = [
            initial_charges.get(elements[i], 0.0)
            for i in independent_indices
        ]
        self._raw_charges = nn.Parameter(
            torch.tensor(initial_charge_values, dtype=torch.float64)
        )

    def get_charges(self):
        charges = torch.zeros(
            self.num_elements,
            device=self._raw_charges.device,
            dtype=self._raw_charges.dtype,
        )
        charges.scatter_(0, self._independent_charge_indices, self._raw_charges)

        if self._enforce_cd_se:
            charges = charges.clone()
            charges[self._se_index] = -charges[self._cd_index]

        return charges

    def forward(self, batch, return_charges=False):
        charges = self.get_charges()

        if self.is_many_body:
            # Many-body path: compute energy, get forces via autograd
            positions = batch.pos.clone().requires_grad_(True)

            V_potential = self.potential.compute_energy(
                positions, batch.edge_index, batch.element_indices,
                self.pair_lookup_table, batch.batch, positions.size(0),
            )

            if self.coulomb:
                # Coulomb energy through autograd too
                source, target = batch.edge_index
                disp = positions[target] - positions[source]
                dist = torch.norm(disp, dim=1)
                q_i = charges[batch.element_indices[source]]
                q_j = charges[batch.element_indices[target]]
                V_coulomb = 0.5 * (q_i * q_j / dist).sum()
                V_total = V_potential + V_coulomb
            else:
                V_total = V_potential

            forces = -torch.autograd.grad(
                V_total, positions,
                create_graph=self.training,
            )[0]
        else:
            # Pair potential path: scalar_force -> project -> scatter
            potential_scalar_force = self.potential.scalar_force(
                batch, batch.pair_indices
            )

            if self.coulomb:
                coulomb_scalar_force = (
                    charges[batch.source_elements]
                    * charges[batch.target_elements]
                    * batch.inverse_distances_sq
                )
                total_scalar_force = potential_scalar_force + coulomb_scalar_force
            else:
                total_scalar_force = potential_scalar_force

            force_vectors = -total_scalar_force.unsqueeze(1) * batch.edge_unit_vectors
            forces = scatter_add(
                force_vectors, batch.edge_index[0],
                dim=0, dim_size=batch.pos.size(0),
            )

        if return_charges:
            return forces, charges
        return forces

    def get_params_dict(self):
        """Return all parameters in user units for display/saving."""
        result = self.potential.get_params_display()
        result['charges'] = self.get_charges().detach().cpu().numpy()
        return result


def initialize_parameters(config, potential_cls, elements,
                          pair_name_to_index, pair_names, xyz_files):
    """Build initial parameter arrays for the potential and charges.

    Returns:
        potential_init: dict of param_name -> np.array (in internal units)
        charges: dict of element -> initial charge value
        frozen_pairs: set of frozen pair names
    """
    num_pairs = len(pair_names)
    param_names = potential_cls.user_param_names()
    param_kinds = potential_cls.param_kinds()

    # Get default values from potential class (by creating a temporary instance)
    # and from config overrides
    defaults_from_config = config.get('potential_defaults', {})

    # RDF peak distances for equilibrium parameter initialization
    rdf_peaks = {}  # pair_name -> peak distance in Bohr
    if config.get('use_smart_sigma', True) and xyz_files:
        rdf_peaks = compute_rdf_peaks(
            pair_names, xyz_files, config['box_size_angstrom']
        )

    # Create a temporary potential to get default parameter values
    temp_pot = potential_cls(num_pairs)
    default_display = temp_pot.get_params_display()  # in user units

    # Build initial arrays (in internal units)
    initial_values = {}
    for pname in param_names:
        kind = param_kinds.get(pname, 'dimensionless')
        # Start with potential's built-in default (already in user units from display)
        default_val = default_display[pname][0]  # scalar default
        if pname in defaults_from_config:
            default_val = defaults_from_config[pname]
        initial_values[pname] = np.full(
            num_pairs, _user_to_internal(default_val, kind)
        )

    # Apply fixed_pairs and initial_guesses
    fixed_pairs_config = config.get('fixed_pairs', {})
    initial_guesses_config = config.get('initial_guesses', {})

    frozen_pair_names = set(fixed_pairs_config.keys())

    for pair_name in pair_names:
        pair_index = pair_name_to_index[pair_name]

        if pair_name in fixed_pairs_config:
            for pname, value in fixed_pairs_config[pair_name].items():
                if pname in initial_values:
                    kind = param_kinds.get(pname, 'dimensionless')
                    initial_values[pname][pair_index] = _user_to_internal(
                        value, kind
                    )
        elif pair_name in initial_guesses_config:
            for pname, value in initial_guesses_config[pair_name].items():
                if pname in initial_values:
                    kind = param_kinds.get(pname, 'dimensionless')
                    initial_values[pname][pair_index] = _user_to_internal(
                        value, kind
                    )
        elif pair_name in rdf_peaks:
            # Map RDF peak to the equilibrium distance parameter
            eq_param = potential_cls.equilibrium_distance_param()
            if eq_param is not None and eq_param in initial_values:
                peak_bohr = rdf_peaks[pair_name]
                peak_ang = peak_bohr * BOHR_TO_ANGSTROM
                param_value_ang = potential_cls.rdf_peak_to_param(peak_ang)
                kind = param_kinds.get(eq_param, 'distance')
                initial_values[eq_param][pair_index] = _user_to_internal(
                    param_value_ang, kind
                )
                print(f"    {pair_name}: RDF peak {peak_ang:.2f} A "
                      f"-> {eq_param} = {param_value_ang:.2f}")

    # Charges
    formal_charges = config.get('formal_charges', {})
    charges = {}
    print(f"\nCharge initialization:")
    for element in elements:
        formal = formal_charges.get(element, 0.0)
        charges[element] = config.get('initial_charges', {}).get(
            element, formal * 0.5
        )
        print(f"    {element}: formal={formal:+.1f}, "
              f"init={charges[element]:+.3f}")

    return initial_values, charges, frozen_pair_names


def save_checkpoint(model, optimizer, epoch, best_rmse, epochs_without_improvement,
                    rmse_history, frozen_mask):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rmse': best_rmse,
        'epochs_without_improvement': epochs_without_improvement,
        'rmse_history': rmse_history,
        'frozen_mask': frozen_mask,
    }, 'checkpoint.pt')


def load_checkpoint(model, optimizer, device):
    if not os.path.exists('checkpoint.pt'):
        return None
    checkpoint = torch.load('checkpoint.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def compute_core_force_rmse(predicted_forces, dft_forces, element_indices,
                            core_mask):
    squared_error = (predicted_forces - dft_forces) ** 2

    if core_mask is not None:
        mask = core_mask[element_indices].unsqueeze(1)
        masked_mse = (squared_error * mask).sum() / (mask.sum() * 3)
        return torch.sqrt(masked_mse + 1e-30)

    return torch.sqrt(squared_error.mean() + 1e-30)


def train(model, train_loader, val_loader, config, frozen_mask=None):
    device = next(model.parameters()).device

    learning_rate = float(config['learning_rate'])
    min_learning_rate = float(config['minimum_learning_rate'])
    patience = int(config['convergence_patience'])
    convergence_threshold = float(config['convergence_threshold']) * FORCE_AU_TO_EV_ANG
    force_weight = float(config['force_weight'])
    neutrality_weight = float(config['neutrality_penalty_weight'])
    enforce_neutrality = config.get('enforce_neutrality', True)

    core_mask = None
    core_elements = set(config.get('core_elements', []))
    if core_elements:
        core_indices = [
            model.element_to_index[e]
            for e in core_elements
            if e in model.element_to_index
        ]
        if core_indices:
            core_mask = torch.zeros(
                model.num_elements, device=device, dtype=torch.float64
            )
            core_mask[core_indices] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_rmse = float('inf')
    epochs_without_improvement = 0
    rmse_history = []
    best_params = None

    checkpoint = load_checkpoint(model, optimizer, device)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_rmse = checkpoint['best_rmse']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        rmse_history = checkpoint.get('rmse_history', [])
        frozen_mask = checkpoint['frozen_mask'].to(device)
        print(f"*** RESUMING FROM CHECKPOINT AT EPOCH {start_epoch} ***")

    # Identify which potential parameter names to save/restore
    pot_param_names = model.potential.parameter_names()

    REACTIVE_WINDOW = 15
    lr_cut_cooldown = 0
    num_lr_cuts = 0

    print(f"\n{'='*60}")
    print("TRAINING")
    print('='*60)
    print(f"LR: {learning_rate:.0e}, Min LR: {min_learning_rate:.0e}, "
          f"Patience: {patience}, Core: {core_elements}")
    print(f"\n{'Epoch':>8} | {'RMSE(eV/A)':>10} | {'Val RMSE':>10} "
          f"| {'LR':>10} | {'Time':>8}")
    print("-" * 60)

    time_limit_seconds = 0.1 * 3600
    training_start_time = time.time()
    epoch = start_epoch

    while True:
        epoch += 1
        elapsed = time.time() - training_start_time

        if elapsed > time_limit_seconds:
            print(f"\nTime limit reached ({time_limit_seconds / 3600:.1f} hours)")
            save_checkpoint(model, optimizer, epoch, best_rmse,
                            epochs_without_improvement, rmse_history, frozen_mask)
            break

        if epochs_without_improvement >= patience:
            print(f"\nConverged (no improvement in {patience} epochs)")
            break

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr <= min_learning_rate * 1.01 and \
                epochs_without_improvement >= patience // 2:
            print(f"\nAt minimum LR with no improvement "
                  f"for {epochs_without_improvement} epochs")
            break

        model.train()
        epoch_rmse_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if enforce_neutrality:
                predicted_forces, charges = model(batch, return_charges=True)
            else:
                predicted_forces = model(batch)
                charges = None

            force_rmse = compute_core_force_rmse(
                predicted_forces, batch.dft_forces,
                batch.element_indices, core_mask,
            )

            loss = force_weight * force_rmse
            if enforce_neutrality and charges is not None:
                net_charge = scatter_add(
                    charges[batch.element_indices], batch.batch, dim=0
                )
                loss = loss + neutrality_weight * torch.mean(net_charge ** 2)

            loss.backward()

            # Apply frozen mask to potential parameters
            if frozen_mask is not None:
                model.potential.apply_frozen_mask(frozen_mask)

            optimizer.step()

            epoch_rmse_sum += force_rmse.detach()
            num_batches += 1

        train_rmse = (epoch_rmse_sum.item() / num_batches) * FORCE_AU_TO_EV_ANG
        rmse_history.append(train_rmse)

        # Reactive LR scheduler: cut LR when RMSE stalls
        if lr_cut_cooldown > 0:
            lr_cut_cooldown -= 1

        if len(rmse_history) >= REACTIVE_WINDOW + 1 and lr_cut_cooldown == 0:
            recent_values = rmse_history[-REACTIVE_WINDOW:]
            baseline = rmse_history[-(REACTIVE_WINDOW + 1)]
            if all(value > baseline for value in recent_values):
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * 0.5, min_learning_rate)
                if new_lr < old_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    num_lr_cuts += 1
                    lr_cut_cooldown = REACTIVE_WINDOW * 2
                    print(f"  ** LR CUT #{num_lr_cuts}: "
                          f"{old_lr:.2e} -> {new_lr:.2e}")

        # Periodic validation
        val_display = "".center(10)
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            val_rmse_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    val_rmse = compute_core_force_rmse(
                        model(batch), batch.dft_forces,
                        batch.element_indices, core_mask,
                    )
                    val_rmse_sum += val_rmse
                    val_batches += 1
            if val_batches > 0:
                val_display = f"{(val_rmse_sum.item() / val_batches) * FORCE_AU_TO_EV_ANG:10.4f}"
            else:
                val_display = "N/A".center(10)

        # Track best parameters (generic for any potential)
        if train_rmse < best_rmse - convergence_threshold:
            best_rmse = train_rmse
            epochs_without_improvement = 0
            best_params = {'epoch': epoch}
            for pname in pot_param_names:
                best_params[pname] = getattr(
                    model.potential, pname
                ).detach().clone()
            best_params['raw_charges'] = model._raw_charges.detach().clone()
        else:
            epochs_without_improvement += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:8d} | {train_rmse:10.4f} | {val_display} "
                  f"| {current_lr:10.2e} | {elapsed / 60:7.1f}m")

        if epoch % 1000 == 0:
            save_checkpoint(model, optimizer, epoch, best_rmse,
                            epochs_without_improvement, rmse_history,
                            frozen_mask)

    if best_params:
        print(f"\nRestoring best parameters from epoch {best_params['epoch']}")
        with torch.no_grad():
            for pname in pot_param_names:
                getattr(model.potential, pname).copy_(best_params[pname])
            model._raw_charges.copy_(best_params['raw_charges'])

    return rmse_history, core_mask


def plot_training_curves(rmse_history, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(rmse_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/A)')
    ax.set_title('Training Loss - Core Atom Force RMSE')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_force_parity(model, loader, core_mask, filepath):
    model.eval()
    all_predicted = []
    all_reference = []

    with torch.no_grad():
        for batch in loader:
            predicted = model(batch)
            if core_mask is not None:
                core_indices = core_mask[batch.element_indices].nonzero(
                    as_tuple=True
                )[0]
                all_predicted.append(predicted[core_indices].cpu().numpy())
                all_reference.append(batch.dft_forces[core_indices].cpu().numpy())
            else:
                all_predicted.append(predicted.cpu().numpy())
                all_reference.append(batch.dft_forces.cpu().numpy())

    predicted_ev_ang = np.concatenate(all_predicted) * FORCE_AU_TO_EV_ANG
    reference_ev_ang = np.concatenate(all_reference) * FORCE_AU_TO_EV_ANG

    predicted_magnitudes = np.linalg.norm(predicted_ev_ang, axis=1)
    reference_magnitudes = np.linalg.norm(reference_ev_ang, axis=1)
    rmse = float(np.sqrt(np.mean(
        (predicted_magnitudes - reference_magnitudes) ** 2
    )))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(
        reference_magnitudes, predicted_magnitudes,
        gridsize=60, cmap='viridis', mincnt=1,
        norm=matplotlib.colors.LogNorm(),
    )
    axis_limit = max(reference_magnitudes.max(), predicted_magnitudes.max()) * 1.05
    ax.plot([0, axis_limit], [0, axis_limit], 'r--', lw=2)
    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.set_xlabel('DFT |F| (eV/A)')
    ax.set_ylabel('Predicted |F| (eV/A)')
    ax.set_title(f'Force Parity - Core Atoms (|F| RMSE = {rmse:.4f} eV/A)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def write_lammps_params(model, pair_names, elements, config, filepath):
    """Write LAMMPS-compatible parameter file, delegating to the potential."""
    potential = model.potential
    params = model.get_params_dict()
    pot_type = config['potential_type']
    cutoff_ang = config['cutoff_angstroms']
    is_many_body = isinstance(potential, ManyBodyPotential)

    with open(filepath, 'w') as f:
        f.write(f"Fitted {pot_type.upper()} + Coulomb Parameters\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if is_many_body:
            # Many-body potentials write their own file format
            base_path = filepath.rsplit('.', 1)[0] if '.' in filepath else filepath
            pair_style, pair_coeff_lines, written_files = \
                potential.write_lammps_file(
                    pair_names, elements, cutoff_ang, base_path)

            if model.coulomb:
                f.write(f"pair_style hybrid/overlay {pair_style} coul/long {cutoff_ang}\n")
                f.write(f"kspace_style ewald 1e-6\n\n")
            else:
                f.write(f"pair_style {pair_style}\n\n")

            # Charges
            f.write("# ATOMIC CHARGES (e)\n")
            for i, elem in enumerate(elements):
                f.write(f"set type {i+1} charge {params['charges'][i]:.6f}  # {elem}\n")
            f.write("\n")

            # Pair coefficients
            f.write(f"# {pot_type.upper()} PAIR COEFFICIENTS\n")
            for line in pair_coeff_lines:
                f.write(line + "\n")
            if model.coulomb:
                f.write(f"pair_coeff * * coul/long\n")

            for wf in written_files:
                print(f"Saved: {wf}")
        else:
            # Pair potential path (unchanged)
            pair_style = potential.lammps_pair_style(cutoff_ang)
            if pair_style is not None:
                f.write(f"pair_style {pair_style}\n")
                f.write(f"pair_modify mix arithmetic\n")
                f.write(f"kspace_style ewald 1e-6\n\n")
            else:
                f.write(f"# Tabulated potential - table files needed\n")
                f.write(f"pair_style hybrid/overlay table linear 5000 coul/long {cutoff_ang}\n")
                f.write(f"kspace_style ewald 1e-6\n\n")

            # Charges
            f.write("# ATOMIC CHARGES (e)\n")
            for i, elem in enumerate(elements):
                f.write(f"set type {i+1} charge {params['charges'][i]:.6f}  # {elem}\n")
            f.write("\n")

            # Pair coefficients
            f.write(f"# {pot_type.upper()} PAIR COEFFICIENTS\n")
            if pair_style is not None:
                coeff_lines = potential.lammps_pair_coeff_lines(
                    pair_names, elements, cutoff_ang
                )
                for line in coeff_lines:
                    f.write(line + "\n")
            else:
                f.write("# Table file pair coefficients\n")
                elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
                for i, pair_name in enumerate(pair_names):
                    e1, e2 = pair_name.split('-')
                    t1, t2 = elem_to_type[e1], elem_to_type[e2]
                    table_fname = f"table_{pot_type}_{pair_name}.dat"
                    f.write(
                        f"pair_coeff {t1} {t2} table "
                        f"{table_fname} {pair_name}\n"
                    )

        f.write("\n# PARAMETERS IN USER UNITS\n")
        header = "# " + f"{'Pair':>8} | " + potential.display_header()
        f.write(header + "\n")
        for i, pair_name in enumerate(pair_names):
            f.write(f"# {pair_name:>8} | {potential.display_row(i)}\n")

    print(f"Saved: {filepath}")

    # Generate table files for tabulated pair potentials
    if not is_many_body and pair_style is None:
        for i, pair_name in enumerate(pair_names):
            table_fname = f"table_{pot_type}_{pair_name}.dat"
            table_contents = potential.generate_table_file(
                pair_name, cutoff_ang
            )
            with open(table_fname, 'w') as f:
                f.write(table_contents)
            print(f"Saved: {table_fname}")


def main():
    global CONFIG

    start_time = time.time()
    print("=" * 60)
    print("Force Field Fitter (Modular Potentials)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    torch.set_default_dtype(torch.float64)

    config_file = sys.argv[1] if len(sys.argv) > 1 else CONFIG_FILE
    CONFIG = load_config(config_file)

    # Resolve potential class
    potential_cls = get_potential(CONFIG['potential_type'])
    print(f"Using potential: {potential_cls.__name__}")

    dataset = DFTDataset(
        CONFIG['datasets'],
        CONFIG['cutoff_angstroms'],
        CONFIG['minimum_distance_angstroms'],
        device,
    )
    elements = dataset.elements

    pair_names = [
        f"{e1}-{e2}"
        for e1, e2 in combinations_with_replacement(elements, 2)
    ]
    pair_name_to_index = {name: i for i, name in enumerate(pair_names)}
    num_pairs = len(pair_names)
    print(f"\nElements: {elements}")
    print(f"Pair types: {num_pairs} ({', '.join(pair_names)})")

    pair_lookup_table = torch.zeros(
        len(elements), len(elements), dtype=torch.long, device=device
    )
    for i in range(len(elements)):
        for j in range(len(elements)):
            low, high = min(i, j), max(i, j)
            canonical_name = f"{elements[low]}-{elements[high]}"
            pair_lookup_table[i, j] = pair_name_to_index[canonical_name]

    # Initialize parameters
    xyz_files = [ds['xyz'] for ds in CONFIG['datasets']]
    initial_values, initial_charges, frozen_pair_names = initialize_parameters(
        CONFIG, potential_cls, elements, pair_name_to_index, pair_names,
        xyz_files,
    )

    # Create potential instance with initialized parameters
    potential = potential_cls(num_pairs, initial_values)
    potential = potential.to(device)

    # Build graphs with fields required by this potential
    is_many_body = isinstance(potential, ManyBodyPotential)
    required_fields = set() if is_many_body else set(potential.required_edge_fields())
    dataset.build_graphs(pair_lookup_table, required_fields,
                         is_many_body=is_many_body)

    # Frozen mask
    frozen_mask = torch.ones(num_pairs, device=device, dtype=torch.float64)
    for pair_name in frozen_pair_names:
        if pair_name in pair_name_to_index:
            frozen_mask[pair_name_to_index[pair_name]] = 0.0
    trainable_pairs = [pn for pn in pair_names if pn not in frozen_pair_names]
    print(f"Frozen pairs: {frozen_pair_names if frozen_pair_names else 'none'}")
    print(f"Trainable pairs: {trainable_pairs}")

    # Data split
    num_total = len(dataset)
    num_val = max(1, int(num_total * CONFIG['validation_split']))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [num_total - num_val, num_val]
    )
    train_loader = DataLoader(
        train_set, batch_size=CONFIG['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG['batch_size'], shuffle=False
    )
    print(f"\nData split: {len(train_set)} train, {len(val_set)} val")

    # Create model
    model = ForceFieldModel(
        elements, pair_lookup_table, num_pairs,
        potential, initial_charges,
        enforce_constraints=CONFIG.get('enforce_element_constraints', True),
        coulomb=CONFIG.get('coulomb', True),
    ).to(device)

    # Print initial parameters
    params = model.get_params_dict()
    user_pnames = potential_cls.user_param_names()
    print(f"\nInitial parameters:")
    header = f"  {'Pair':>8} | " + potential.display_header() + f" | {'Frozen':>6}"
    print(header)
    for i, pair_name in enumerate(pair_names):
        frozen_label = "YES" if pair_name in frozen_pair_names else "no"
        print(f"  {pair_name:>8} | {potential.display_row(i)} | {frozen_label:>6}")
    print(f"\n  {'Element':>8} | {'Charge (e)':>10}")
    for i, element in enumerate(elements):
        print(f"  {element:>8} | {params['charges'][i]:+10.4f}")

    # Train
    rmse_history, core_mask = train(
        model, train_loader, val_loader, CONFIG, frozen_mask=frozen_mask
    )

    # Print final parameters
    params = model.get_params_dict()
    print(f"\n{'='*60}")
    print("FINAL PARAMETERS")
    print('='*60)
    print(f"\n  {'Pair':>8} | " + potential.display_header())
    for i, pair_name in enumerate(pair_names):
        if pair_name in frozen_pair_names:
            continue
        print(f"  {pair_name:>8} | {potential.display_row(i)}")
    print(f"\n  {'Element':>8} | {'Charge (e)':>10}")
    for i, element in enumerate(elements):
        print(f"  {element:>8} | {params['charges'][i]:+10.6f}")

    # Save outputs
    plot_training_curves(rmse_history, 'training_curves.png')
    plot_force_parity(model, train_loader, core_mask, 'force_parity.png')
    write_lammps_params(model, pair_names, elements, CONFIG, 'lammps_params.txt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'elements': elements,
        'pair_names': pair_names,
        'frozen_pairs': frozen_pair_names,
        'potential_type': CONFIG['potential_type'],
        'config': CONFIG,
        'rmse_history': rmse_history,
    }, 'final_model.pt')
    print(f"Saved: final_model.pt")

    print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} min")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
