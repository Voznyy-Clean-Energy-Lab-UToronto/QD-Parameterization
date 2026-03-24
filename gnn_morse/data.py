import os
import numpy as np
import torch
import ase.io
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from .utils import (
    ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE, FORCE_AU_TO_EV_ANG,
    canonical_pair,
)

# ── Data readers ─────────────────────────────────────────────────────

def read_extxyz(filepath, first_n=None):
    frames = ase.io.read(filepath, index=':', format='extxyz')
    symbols = frames[0].get_chemical_symbols()
    print(f"  Frames: {len(frames)}, Atoms: {len(symbols)}")

    if first_n and first_n < len(frames):
        frames = frames[:first_n]
        print(f"  Using first {first_n} frames")

    positions, forces, energies = [], [], []
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        try:
            ff = frame.get_forces()
            forces.append(ff / FORCE_AU_TO_EV_ANG)
        except Exception:
            forces.append(None)
        energies.append(float(frame.info['energy']) if 'energy' in frame.info else None)

    return symbols, positions, forces, energies


def read_lammps_dump(filepath, first_n=None):
    positions_list, forces_list = [], []
    atom_symbols = None

    with open(filepath) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'ITEM: TIMESTEP' not in line:
                continue
            f.readline()  # timestep
            f.readline()  # NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())
            f.readline()  # BOX BOUNDS
            f.readline(); f.readline(); f.readline()
            cols = f.readline().strip().replace('ITEM: ATOMS', '').split()
            col_idx = {name: i for i, name in enumerate(cols)}

            px = col_idx.get('xu', col_idx.get('x'))
            py = col_idx.get('yu', col_idx.get('y'))
            pz = col_idx.get('zu', col_idx.get('z'))
            has_forces = all(k in col_idx for k in ('fx', 'fy', 'fz'))
            has_id = 'id' in col_idx

            pos = np.empty((n_atoms, 3))
            frc = np.empty((n_atoms, 3)) if has_forces else None
            elems = []
            ids = np.empty(n_atoms, dtype=int)

            for j in range(n_atoms):
                tokens = f.readline().split()
                pos[j] = [float(tokens[px]), float(tokens[py]), float(tokens[pz])]
                if 'element' in col_idx:
                    elems.append(tokens[col_idx['element']])
                ids[j] = int(tokens[col_idx['id']]) if has_id else j
                if has_forces:
                    frc[j] = [float(tokens[col_idx['fx']]),
                              float(tokens[col_idx['fy']]),
                              float(tokens[col_idx['fz']])]

            order = np.argsort(ids)
            pos = pos[order]
            if has_forces:
                frc = frc[order]
            if elems:
                elems = [elems[i] for i in order]

            if atom_symbols is None:
                atom_symbols = elems
            positions_list.append(pos * ANGSTROM_TO_BOHR)
            forces_list.append(frc / FORCE_AU_TO_EV_ANG if has_forces and frc is not None else None)

    print(f"  Read {len(positions_list)} frames, {len(atom_symbols)} atoms")
    if first_n and first_n < len(positions_list):
        positions_list = positions_list[:first_n]
        forces_list = forces_list[:first_n]
        print(f"  Using first {first_n} frames")

    energies = [None] * len(positions_list)
    return atom_symbols, positions_list, forces_list, energies


# ── Gap detection ────────────────────────────────────────────────────

def detect_gap_cutoff(dist_matrix, max_neighbors=20):
    dists = dist_matrix.copy()
    dists[(dists < 1e-10) | ~np.isfinite(dists)] = np.inf

    sorted_dists = np.sort(dists, axis=1)
    max_neighbors_to_check = min(max_neighbors + 1, sorted_dists.shape[1])
    nearest = sorted_dists[:, :max_neighbors_to_check]

    n_finite_per_atom = np.sum(np.isfinite(nearest), axis=1)
    has_gap = n_finite_per_atom >= 2

    gaps = np.diff(nearest, axis=1)
    gaps[~np.isfinite(gaps)] = 0

    max_gap_pos = np.argmax(gaps, axis=1)
    row_idx = np.arange(len(nearest))
    cutoffs = 0.5 * (nearest[row_idx, max_gap_pos] + nearest[row_idx, max_gap_pos + 1])
    cutoffs[~has_gap | ~np.isfinite(cutoffs)] = np.inf

    finite_cutoffs = cutoffs[np.isfinite(cutoffs)]
    median = float(np.median(finite_cutoffs)) if len(finite_cutoffs) > 0 else float('inf')

    return median, cutoffs


# ── KNN edge builder ────────────────────────────────────────────────

def build_knn_edges(positions_bohr, atom_symbols, knn_config):
    inorganic = set(knn_config['inorganic_elements'])
    k_cross = knn_config['k_cross']
    max_neighbors = knn_config.get('max_gap_neighbors', 20)
    ligand_cutoff_bohr = knn_config.get('ligand_cutoff', 4.0) * ANGSTROM_TO_BOHR

    pos = np.asarray(positions_bohr)
    symbols = np.array(atom_symbols)

    inorg_by_elem = {}
    for elem in sorted(inorganic):
        mask = symbols == elem
        if mask.any():
            inorg_by_elem[elem] = np.where(mask)[0]
    ligand_idx = np.where(~np.isin(symbols, list(inorganic)))[0]
    inorg_elems = sorted(inorg_by_elem.keys())

    edge_arrays = []
    gap_cuts_bohr = {}

    def _collect_undirected(global_src, global_tgt):
        lo = np.minimum(global_src, global_tgt)
        hi = np.maximum(global_src, global_tgt)
        edge_arrays.append(np.column_stack([lo, hi]))

    # Cross-species: k nearest within gap cutoff, both directions
    for i, elem_a in enumerate(inorg_elems):
        for elem_b in inorg_elems[i + 1:]:
            idx_a, idx_b = inorg_by_elem[elem_a], inorg_by_elem[elem_b]
            dists = cdist(pos[idx_a], pos[idx_b])

            cut_ab, _ = detect_gap_cutoff(dists, max_neighbors)
            cut_ba, _ = detect_gap_cutoff(dists.T, max_neighbors)
            gap_cut = 0.5 * (cut_ab + cut_ba)
            gap_cuts_bohr[canonical_pair(elem_a, elem_b)] = gap_cut

            for source_idx, target_idx, d in [(idx_a, idx_b, dists),
                                               (idx_b, idx_a, dists.T)]:
                k = min(k_cross, d.shape[1])
                knn_cols = np.argsort(d, axis=1)[:, :k]
                knn_dists = np.take_along_axis(d, knn_cols, axis=1)
                rows, cols = np.where(knn_dists <= gap_cut)
                if len(rows) > 0:
                    _collect_undirected(source_idx[rows],
                                        target_idx[knn_cols[rows, cols]])

    # Same-species: all pairs within gap cutoff (no k limit)
    for elem in inorg_elems:
        idx = inorg_by_elem[elem]
        if len(idx) < 2:
            continue
        dists = cdist(pos[idx], pos[idx])
        np.fill_diagonal(dists, np.inf)

        gap_cut, _ = detect_gap_cutoff(dists, max_neighbors)
        gap_cuts_bohr[canonical_pair(elem, elem)] = gap_cut

        li, lj = np.where(np.triu(dists <= gap_cut, k=1))
        if len(li) > 0:
            edge_arrays.append(np.column_stack([idx[li], idx[lj]]))

    # Inorganic-ligand: fixed distance cutoff
    if len(ligand_idx) > 0:
        all_inorg_idx = np.concatenate(list(inorg_by_elem.values()))
        dists = cdist(pos[all_inorg_idx], pos[ligand_idx])
        ii, jj = np.where(dists < ligand_cutoff_bohr)
        if len(ii) > 0:
            _collect_undirected(all_inorg_idx[ii], ligand_idx[jj])

        ligand_elems = sorted(set(symbols[ligand_idx]))
        for ie in inorg_elems:
            for le in ligand_elems:
                gap_cuts_bohr[canonical_pair(ie, le)] = ligand_cutoff_bohr

    # Deduplicate and symmetrize to directed edges
    if edge_arrays:
        unique_edges = np.unique(np.vstack(edge_arrays), axis=0)
    else:
        unique_edges = np.empty((0, 2), dtype=np.int64)

    src = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]]).astype(np.int64)
    tgt = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]]).astype(np.int64)
    gap_cuts_angstrom = {pair: cut * BOHR_TO_ANGSTROM
                         for pair, cut in gap_cuts_bohr.items()}

    return src, tgt, gap_cuts_angstrom


# ── Dataset ──────────────────────────────────────────────────────────

class DFTDataset:

    def __init__(self, dataset_configs, knn_config, first_n_frames=None):
        self.frame_data = []      # [(symbols, pos_bohr), ...]
        self.forces_data = []     # [forces_au or None, ...]
        self.energies_data = []   # [energy_ev or None, ...]
        all_elements = set()

        print(f"\n{'='*60}")
        print("LOADING DATA")
        print('='*60)

        for ds in dataset_configs:
            name = ds.get('name', ds['xyz'])
            fmt = ds.get('format', 'extxyz')
            ds_first_n = ds.get('first_n_frames', first_n_frames)
            print(f"\n{name} (format: {fmt})")

            if fmt == 'lammps':
                symbols, pos_list, frc_list, eng_list = read_lammps_dump(
                    ds['xyz'], first_n=ds_first_n)
            else:
                symbols, pos_list, frc_list, eng_list = read_extxyz(
                    ds['xyz'], first_n=ds_first_n)

            all_elements.update(symbols)
            self.frame_data.extend([(symbols, p) for p in pos_list])
            self.forces_data.extend(frc_list)
            self.energies_data.extend(eng_list)

        self.elements = sorted(all_elements)
        self.element_to_index = {e: i for i, e in enumerate(self.elements)}
        self.knn_config = knn_config

        # Pair names (canonical, sorted)
        self.pair_names = [
            f"{e1}-{e2}"
            for e1, e2 in combinations_with_replacement(self.elements, 2)
        ]
        self.pair_name_to_index = {pn: i for i, pn in enumerate(self.pair_names)}

        # Build pair lookup table (element_i, element_j) -> pair_index
        ne = len(self.elements)
        self.pair_lookup = np.zeros((ne, ne), dtype=np.int64)
        for i in range(ne):
            for j in range(ne):
                lo, hi = min(i, j), max(i, j)
                self.pair_lookup[i, j] = self.pair_name_to_index[
                    f"{self.elements[lo]}-{self.elements[hi]}"]

        # Process energies (relative, in Hartree)
        valid_e = [e for e in self.energies_data if e is not None]
        if valid_e:
            self.mean_energy_ev = float(np.mean(valid_e))
            self.energies_au = [
                (e - self.mean_energy_ev) * EV_TO_HARTREE if e is not None else None
                for e in self.energies_data
            ]
            print(f"\nEnergies: {len(valid_e)} valid, mean = {self.mean_energy_ev:.4f} eV")
        else:
            self.mean_energy_ev = 0.0
            self.energies_au = [None] * len(self.energies_data)
            print("\nNo energies in dataset")

        print(f"Total: {len(self.frame_data)} frames, Elements: {self.elements}")
        print(f"Pairs ({len(self.pair_names)}): {', '.join(self.pair_names)}")

    def build_graphs(self):
        print("\nBuilding KNN graphs...", end=" ", flush=True)
        self.graphs = []
        pair_lookup_t = torch.tensor(self.pair_lookup, dtype=torch.long)

        self.gap_cuts_angstrom = None  # store from first frame

        for i, (symbols, positions_bohr) in enumerate(self.frame_data):
            # KNN edges (numpy, CPU)
            src, tgt, gap_cuts_ang = build_knn_edges(
                positions_bohr, symbols, self.knn_config)
            if self.gap_cuts_angstrom is None:
                self.gap_cuts_angstrom = gap_cuts_ang

            if len(src) == 0:
                print(f"\nWarning: frame {i} has 0 edges!")
                continue

            pos = torch.tensor(positions_bohr, dtype=torch.float64)
            src_t = torch.tensor(src, dtype=torch.long)
            tgt_t = torch.tensor(tgt, dtype=torch.long)

            # Distances and unit vectors
            disp = pos[tgt_t] - pos[src_t]
            dist = torch.norm(disp, dim=1)
            unit_vec = disp / dist.unsqueeze(1)

            # Element indices
            elem_idx = torch.tensor(
                [self.element_to_index[s] for s in symbols],
                dtype=torch.long)

            # Pair indices
            src_elem = elem_idx[src_t]
            tgt_elem = elem_idx[tgt_t]
            pair_indices = pair_lookup_t[src_elem, tgt_elem]

            # DFT forces
            frc = self.forces_data[i]
            dft_forces = torch.tensor(frc, dtype=torch.float64) if frc is not None else torch.zeros_like(pos)

            # DFT energy
            eng = self.energies_au[i]
            dft_energy = torch.tensor([eng], dtype=torch.float64) if eng is not None else None

            self.graphs.append(Data(
                pos=pos,
                edge_index=torch.stack([src_t, tgt_t]),
                distances=dist,
                edge_unit_vectors=unit_vec,
                element_indices=elem_idx,
                pair_indices=pair_indices,
                dft_forces=dft_forces,
                dft_energy=dft_energy,
            ))

        print(f"done ({len(self.graphs)} graphs)")

        # Print edge statistics for first frame
        if self.graphs:
            g = self.graphs[0]
            print("\nEdge statistics (frame 0):")
            for pi, pn in enumerate(self.pair_names):
                mask = g.pair_indices == pi
                count = mask.sum().item()
                if count > 0:
                    d = g.distances[mask] * BOHR_TO_ANGSTROM
                    print(f"  {pn:>8}: {count:5d} edges, "
                          f"dist = {d.min():.2f} - {d.max():.2f} A "
                          f"(mean {d.mean():.2f})")
                else:
                    print(f"  {pn:>8}:     0 edges")
