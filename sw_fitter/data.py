import math
from collections import Counter
from itertools import combinations, combinations_with_replacement

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import cdist

from .utils import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    FORCE_AU_TO_EV_ANG,
    canonical_pair,
    canonical_triplet,
)

ZHOU_B = 1.116149    # initial B 
SIGMA_RATIO = 1.28094    # r0 / sigma  (location of V2 minimum)
CUTOFF_RATIO = 1.953387  # cutoff / sigma  (SW 'a' parameter)
P = 4.0  # (sigma/r)^p exponent

BOLTZMANN_EV = 8.617333e-5  # eV/K

BOND_MAX_ANG = 5.5

INORGANIC = {"Cd", "Se"}


def read_trajectory(filepath, fmt="extxyz", first_n=None, skip_n=None):
    ase_format = "lammps-dump-text" if fmt == "lammps" else "extxyz"
    frames = ase.io.read(filepath, index=":", format=ase_format)
    if skip_n:
        frames = frames[skip_n:]
    if first_n:
        frames = frames[:first_n]

    symbols = frames[0].get_chemical_symbols()
    positions, forces = [], []
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        frame_forces = frame.arrays.get("forces")
        if frame_forces is None and frame.calc is not None:
            frame_forces = frame.calc.results.get("forces")
        forces.append(frame_forces / FORCE_AU_TO_EV_ANG if frame_forces is not None else None)

    print(f"  {len(frames)} frames, {len(symbols)} atoms")
    return symbols, positions, forces


def is_scoped_pair(element_a, element_b):
    both_inorganic = element_a in INORGANIC and element_b in INORGANIC
    inorganic_O = (element_a in INORGANIC and element_b == "O") or (
        element_b in INORGANIC and element_a == "O"
    )
    return both_inorganic or inorganic_O


def _compute_A_from_B_scalar(B, sigma, r0, cutoff):
    bracket = B * (sigma / r0) ** P - 1.0
    decay = math.exp(sigma / (r0 - cutoff))
    return -1.0 / (bracket * decay)


def _v2_scalar(r, eps, A, B, sigma, cutoff):
    if r >= cutoff:
        return 0.0
    return eps * A * (B * (sigma / r) ** P - 1.0) * math.exp(sigma / (r - cutoff))


def _d2v2_shape_factor(B, sigma, r0, cutoff, dr=1e-5):
    A = _compute_A_from_B_scalar(B, sigma, r0, cutoff)
    v_p = _v2_scalar(r0 + dr, 1.0, A, B, sigma, cutoff)
    v_m = _v2_scalar(r0 - dr, 1.0, A, B, sigma, cutoff)
    v_0 = _v2_scalar(r0,      1.0, A, B, sigma, cutoff)
    return (v_p - 2.0 * v_0 + v_m) / dr ** 2


def bond_geometry_from_rdf(sampled_positions_bohr, atoms_a, atoms_b, same_element, temperature):
    bond_max_bohr = BOND_MAX_ANG * ANGSTROM_TO_BOHR
    bin_edges = np.linspace(1.5 * ANGSTROM_TO_BOHR, bond_max_bohr, 100)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    histogram = np.zeros(len(bin_centres))
    coordination = 0.0
    for positions in sampled_positions_bohr:
        distances = cdist(positions[atoms_a], positions[atoms_b])
        if same_element:
            np.fill_diagonal(distances, np.inf)
        histogram += np.histogram(distances, bins=bin_edges)[0]
        coordination += (distances < bond_max_bohr).sum() / len(atoms_a)

    g = histogram / np.maximum(bin_centres ** 2, 1e-12)
    g = np.convolve(g, np.ones(3) / 3, mode="same")

    if g.max() < 1e-10:
        return None

    r0_bohr = float(bin_centres[np.argmax(g)])

    after_peak = (bin_centres > r0_bohr) & (bin_centres < r0_bohr + 2.0 * ANGSTROM_TO_BOHR)
    if after_peak.sum() < 2:
        return None
    first_min_bohr = float(bin_centres[after_peak][np.argmin(g[after_peak])])

    coordination = coordination / len(sampled_positions_bohr)

    shell_dists = []
    for positions in sampled_positions_bohr:
        distances = cdist(positions[atoms_a], positions[atoms_b])
        if same_element:
            np.fill_diagonal(distances, np.inf)
        in_shell = distances[(distances > 0.5 * ANGSTROM_TO_BOHR) & (distances < first_min_bohr)]
        shell_dists.extend(in_shell.flatten().tolist())

    var_r_bohr_sq = float(np.var(shell_dists)) if len(shell_dists) > 10 else 1e-3

    return r0_bohr, first_min_bohr, coordination, var_r_bohr_sq


def measure_pair_scales(sampled_positions_bohr, symbols, elements, temperature):
    symbols = np.array(symbols)
    atoms_of_element = {e: np.where(symbols == e)[0] for e in elements}

    scales = {field: {} for field in ("r0", "sigma", "cutoff", "B_init", "eps_init")}
    kT_ev = BOLTZMANN_EV * temperature

    print("\n2-body scales (r0 measured from RDF; eps_init from bond-length equipartition):")
    for element_a, element_b in combinations_with_replacement(elements, 2):
        if not is_scoped_pair(element_a, element_b):
            continue
        atoms_a = atoms_of_element.get(element_a, np.array([], dtype=int))
        atoms_b = atoms_of_element.get(element_b, np.array([], dtype=int))
        if len(atoms_a) == 0 or len(atoms_b) == 0:
            continue

        geometry = bond_geometry_from_rdf(
            sampled_positions_bohr, atoms_a, atoms_b,
            same_element=(element_a == element_b),
            temperature=temperature,
        )
        if geometry is None:
            continue
        r0_bohr, first_min_bohr, coordination, var_r_bohr_sq = geometry
        bond = canonical_pair(element_a, element_b)

        if coordination < 0.05 or not (first_min_bohr > r0_bohr > 0):
            print(f"  {bond:>8}: no clear bond (coordination={coordination:.2f}) — skipped")
            continue

        sigma_bohr = r0_bohr / SIGMA_RATIO
        cutoff_bohr = CUTOFF_RATIO * sigma_bohr

        # eps init: equipartition of radial motion.
        # kT = k_eff * Var(r) => k_eff = kT / Var(r) [eV/Å²]
        # k_eff = eps * shape_factor  => eps = k_eff / shape_factor
        # shape_factor = d²V2/dr²|_r0 with eps=1, A from B_init, in Bohr units
        shape_f = _d2v2_shape_factor(ZHOU_B, sigma_bohr, r0_bohr, cutoff_bohr)
        kT_hartree = kT_ev / 27.2114  # eV → Hartree
        k_eff_hartree_per_bohr_sq = kT_hartree / var_r_bohr_sq
        eps_init_hartree = k_eff_hartree_per_bohr_sq / shape_f

        scales["r0"][bond] = r0_bohr
        scales["sigma"][bond] = sigma_bohr
        scales["cutoff"][bond] = cutoff_bohr
        scales["B_init"][bond] = ZHOU_B
        scales["eps_init"][bond] = eps_init_hartree

        eps_init_ev = eps_init_hartree * 27.2114
        print(
            f"  {bond:>8}: r0={r0_bohr * BOHR_TO_ANGSTROM:.3f} Å  "
            f"sigma={sigma_bohr * BOHR_TO_ANGSTROM:.3f} Å  "
            f"cutoff={cutoff_bohr * BOHR_TO_ANGSTROM:.3f} Å  "
            f"Var(r)^0.5={var_r_bohr_sq**0.5 * BOHR_TO_ANGSTROM:.3f} Å  "
            f"eps_init={eps_init_ev:.3f} eV"
        )
    return scales


def enumerate_triplet_types(elements, scoped_bonds):
    names = set()
    for centre in elements:
        if centre not in INORGANIC:
            continue
        legs = [e for e in elements if canonical_pair(centre, e) in scoped_bonds]
        for leg_a, leg_b in combinations_with_replacement(legs, 2):
            names.add(canonical_triplet(centre, leg_a, leg_b))
    return sorted(names)


def build_edges(positions_bohr, symbols, cutoffs_bohr):
    if not cutoffs_bohr:
        return {}
    positions_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM
    search_radius = max(cutoffs_bohr.values()) * BOHR_TO_ANGSTROM
    atoms = Atoms(symbols=list(symbols), positions=positions_ang)
    first, second, distance_ang = neighbor_list("ijd", atoms, search_radius, self_interaction=False)

    symbols_arr = np.array(symbols)
    once = first < second
    edges = {}
    for atom_i, atom_j, dist_bohr in zip(
        first[once], second[once], distance_ang[once] / BOHR_TO_ANGSTROM
    ):
        bond = canonical_pair(symbols_arr[atom_i], symbols_arr[atom_j])
        if bond in cutoffs_bohr and dist_bohr <= cutoffs_bohr[bond]:
            i_list, j_list = edges.setdefault(bond, ([], []))
            i_list.append(atom_i)
            j_list.append(atom_j)

    return {
        bond: (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long))
        for bond, (i, j) in edges.items()
    }


def build_triplets(edges, symbols, triplet_names):
    neighbours = {}
    for atom_i_t, atom_j_t in edges.values():
        for atom_i, atom_j in zip(atom_i_t.tolist(), atom_j_t.tolist()):
            neighbours.setdefault(atom_i, []).append(atom_j)
            neighbours.setdefault(atom_j, []).append(atom_i)

    symbols_arr = np.array(symbols)
    triplet_names_set = set(triplet_names)
    triplets = {}
    for centre, centre_neighbours in neighbours.items():
        for atom_a, atom_b in combinations(centre_neighbours, 2):
            name = canonical_triplet(symbols_arr[centre], symbols_arr[atom_a], symbols_arr[atom_b])
            if name not in triplet_names_set:
                continue
            if symbols_arr[atom_b] < symbols_arr[atom_a]:
                atom_a, atom_b = atom_b, atom_a
            c_list, a_list, b_list = triplets.setdefault(name, ([], [], []))
            c_list.append(centre)
            a_list.append(atom_a)
            b_list.append(atom_b)

    return {
        name: (
            torch.tensor(c, dtype=torch.long),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
        )
        for name, (c, a, b) in triplets.items()
    }


def make_batch(frames, atoms_per_frame):
    positions = torch.cat([f["positions"] for f in frames])
    dft_forces = torch.cat([f["dft_forces"] for f in frames])
    fit_mask = torch.cat([f["fit_mask"] for f in frames])

    edges, triplets = {}, {}
    offset = 0
    for frame in frames:
        for bond, (atom_i, atom_j) in frame["edges"].items():
            i_list, j_list = edges.setdefault(bond, ([], []))
            i_list.append(atom_i + offset)
            j_list.append(atom_j + offset)
        for name, (centre, atom_a, atom_b) in frame["triplets"].items():
            c_list, a_list, b_list = triplets.setdefault(name, ([], [], []))
            c_list.append(centre + offset)
            a_list.append(atom_a + offset)
            b_list.append(atom_b + offset)
        offset += atoms_per_frame

    edges = {bond: (torch.cat(i), torch.cat(j)) for bond, (i, j) in edges.items()}
    triplets = {
        name: (torch.cat(c), torch.cat(a), torch.cat(b))
        for name, (c, a, b) in triplets.items()
    }
    return {
        "positions": positions,
        "dft_forces": dft_forces,
        "fit_mask": fit_mask,
        "edges": edges,
        "triplets": triplets,
    }


class DFTDataset:
    def __init__(self, dataset_configs, first_n_frames=None):
        print(f"\n{'=' * 60}\nLOADING DATA\n{'=' * 60}")
        self.symbols = None
        self.positions = []
        self.forces = []
        for dataset in dataset_configs:
            print(f"\n{dataset.get('name', dataset['xyz'])}")
            symbols, positions, forces = read_trajectory(
                dataset["xyz"],
                fmt=dataset.get("format", "extxyz"),
                first_n=dataset.get("first_n_frames", first_n_frames),
                skip_n=dataset.get("skip_frames"),
            )
            self.symbols = symbols
            self.positions += positions
            self.forces += forces

        self.atoms_per_frame = len(self.symbols)
        self.elements = sorted(set(self.symbols))
        self.fit_mask = np.array([e in INORGANIC for e in self.symbols])

        counts = Counter(self.symbols)
        print("\nElements (" + ", ".join(f"{e}={counts[e]}" for e in self.elements) + ")")
        print(f"{len(self.positions)} frames, {self.atoms_per_frame} atoms each")

    def build_graphs(self, temperature=650):
        sample = self.positions[:: max(1, len(self.positions) // 50)]
        sampled = [np.asarray(p) for p in sample]
        self.scales = measure_pair_scales(sampled, self.symbols, self.elements, temperature)
        self.cutoffs_bohr = self.scales["cutoff"]
        self.triplet_type_names = enumerate_triplet_types(self.elements, self.cutoffs_bohr)
        print(
            f"Scoped bonds: {len(self.cutoffs_bohr)}, "
            f"triplet types: {len(self.triplet_type_names)}"
        )

        fit_mask = torch.tensor(self.fit_mask, dtype=torch.bool)
        print("\nBuilding graphs...")
        self.graphs = []
        for positions_bohr, forces_au in zip(self.positions, self.forces):
            edges = build_edges(positions_bohr, self.symbols, self.cutoffs_bohr)
            if not edges:
                continue
            triplets = build_triplets(edges, self.symbols, self.triplet_type_names)
            positions = torch.tensor(np.asarray(positions_bohr), dtype=torch.float64)
            dft_forces = (
                torch.tensor(forces_au, dtype=torch.float64)
                if forces_au is not None
                else torch.zeros_like(positions)
            )
            self.graphs.append({
                "positions": positions,
                "dft_forces": dft_forces,
                "fit_mask": fit_mask,
                "edges": edges,
                "triplets": triplets,
            })
        print(f"done ({len(self.graphs)} graphs)")
