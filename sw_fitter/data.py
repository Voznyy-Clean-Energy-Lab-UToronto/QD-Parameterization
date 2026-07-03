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
    EV_TO_HARTREE,
    FORCE_AU_TO_EV_ANG,
    HARTREE_TO_EV,
    canonical_pair,
    canonical_triplet,
)

ZHOU_B = 1.116149    # initial B — will be optimized per bond
SIGMA_RATIO = 1.28094    # r0 / sigma  (location of V2 minimum) FIX LATER
CUTOFF_RATIO = 1.953387  # cutoff / sigma  (SW 'a' parameter) FIX LATER
P = 4.0  # (sigma/r)^p exponent

BOLTZMANN_EV = 8.617333e-5  # eV/K
BOND_MAX_ANG = 5.5
INORGANIC = {"Cd", "Se"} #Make sure not Cd Se only
_QD_ELEMENTS = {"Cd", "Se", "Zn", "S", "Pb", "Te"}


def _chemical_formula(symbols):
    counts = Counter(symbols)
    qd = sorted(e for e in counts if e in _QD_ELEMENTS)
    other = sorted(e for e in counts if e not in _QD_ELEMENTS)
    return "".join(f"{e}{counts[e]}" for e in qd + other)


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


def bond_geometry_from_rdf(sampled_positions_bohr, atoms_a, atoms_b, same_element):
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

    # First minimum: look after the peak up to +2 Å
    after_peak = (bin_centres > r0_bohr) & (bin_centres < r0_bohr + 2.0 * ANGSTROM_TO_BOHR)
    if after_peak.sum() < 2:
        return None
    first_min_bohr = float(bin_centres[after_peak][np.argmin(g[after_peak])])

    coordination = coordination / len(sampled_positions_bohr)

    # Variance of first-shell bond lengths (up to the first minimum)
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

        # eps_init: match equipartition stiffness kT/Var(r) to the SW well curvature (see guide).
        shape_f = _d2v2_shape_factor(ZHOU_B, sigma_bohr, r0_bohr, cutoff_bohr)
        kT_hartree = kT_ev * EV_TO_HARTREE
        k_eff_hartree_per_bohr_sq = kT_hartree / var_r_bohr_sq
        eps_init_hartree = k_eff_hartree_per_bohr_sq / shape_f

        scales["r0"][bond] = r0_bohr
        scales["sigma"][bond] = sigma_bohr
        scales["cutoff"][bond] = cutoff_bohr
        scales["B_init"][bond] = ZHOU_B
        scales["eps_init"][bond] = eps_init_hartree

        eps_init_ev = eps_init_hartree * HARTREE_TO_EV
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


def make_batch(frames):
    positions = torch.cat([f["positions"] for f in frames])
    dft_forces = torch.cat([f["dft_forces"] for f in frames])
    fit_mask = torch.cat([f["fit_mask"] for f in frames])

    edges, triplets = {}, {}
    offset = 0
    for frame in frames:
        n_atoms = frame["positions"].shape[0]
        for bond, (atom_i, atom_j) in frame["edges"].items():
            i_list, j_list = edges.setdefault(bond, ([], []))
            i_list.append(atom_i + offset)
            j_list.append(atom_j + offset)
        for name, (centre, atom_a, atom_b) in frame["triplets"].items():
            c_list, a_list, b_list = triplets.setdefault(name, ([], [], []))
            c_list.append(centre + offset)
            a_list.append(atom_a + offset)
            b_list.append(atom_b + offset)
        offset += n_atoms

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

        # Kept separate because different QDs can have different atom counts.
        self._datasets = []
        for cfg in dataset_configs:
            print(f"\n{cfg.get('name', cfg['xyz'])}")
            symbols, positions, forces = read_trajectory(
                cfg["xyz"],
                fmt=cfg.get("format", "extxyz"),
                first_n=cfg.get("first_n_frames", first_n_frames),
                skip_n=cfg.get("skip_frames"),
            )
            self._datasets.append({
                "name":      cfg.get("name", ""),
                "symbols":   symbols,
                "positions": positions,
                "forces":    forces,
            })

        self.elements = sorted({e for ds in self._datasets for e in ds["symbols"]})

        # chemical_formula names the output .sw file. A single dataset gets its
        # exact formula (e.g. Cd68Se55...); a multi-QD run gets the shared framework
        # elements plus a "_universal" tag (e.g. CdSe_universal).
        # REMOVE UNIVERSAL - SMARTER NAMING?
        if len(self._datasets) == 1:
            self.chemical_formula = _chemical_formula(self._datasets[0]["symbols"])
        else:
            framework_elems = sorted(e for e in self.elements if e in _QD_ELEMENTS)
            self.chemical_formula = "".join(framework_elems) + "_universal"

        total_frames = sum(len(ds["positions"]) for ds in self._datasets)
        print(f"\nElements: {self.elements}")
        print(f"{len(self._datasets)} dataset(s), {total_frames} frames total")

    def build_graphs(self, temperature=650): #this is stupid
        ref = self._datasets[0]
        ref_sample = ref["positions"][:: max(1, len(ref["positions"]) // 50)]
        self.scales = measure_pair_scales(
            [np.asarray(p) for p in ref_sample],
            ref["symbols"],
            sorted(set(ref["symbols"])),
            temperature,
        )
        self.cutoffs_bohr = self.scales["cutoff"]
        self.triplet_type_names = enumerate_triplet_types(self.elements, self.cutoffs_bohr)
        print(
            f"Scoped bonds: {len(self.cutoffs_bohr)}, "
            f"triplet types: {len(self.triplet_type_names)}"
        )

        print("\nBuilding graphs...")
        self.graphs = []
        self.graphs_per_dataset = []
        for ds in self._datasets:
            symbols   = ds["symbols"]
            fit_mask  = torch.tensor(
                [e in INORGANIC for e in symbols], dtype=torch.bool
            )
            qd_graphs = []
            for positions_bohr, forces_au in zip(ds["positions"], ds["forces"]):
                edges = build_edges(positions_bohr, symbols, self.cutoffs_bohr)
                if not edges:
                    continue
                triplets = build_triplets(edges, symbols, self.triplet_type_names)
                positions_t = torch.tensor(np.asarray(positions_bohr), dtype=torch.float64)
                dft_forces = (
                    torch.tensor(forces_au, dtype=torch.float64)
                    if forces_au is not None
                    else torch.zeros_like(positions_t)
                )
                graph = {
                    "positions":  positions_t,
                    "dft_forces": dft_forces,
                    "fit_mask":   fit_mask,
                    "edges":      edges,
                    "triplets":   triplets,
                }
                self.graphs.append(graph)
                qd_graphs.append(graph)
            self.graphs_per_dataset.append(qd_graphs)
        print(f"done ({len(self.graphs)} graphs)")
