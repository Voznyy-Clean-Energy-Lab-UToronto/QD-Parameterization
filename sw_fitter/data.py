from itertools import combinations, combinations_with_replacement
from collections import Counter

import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import cdist

from .utils import (ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG,
                    canonical_pair, canonical_triplet)


# The II-VI Stillinger-Weber 2-body shape (Zhou et al., PRB 88, 085309, 2013).
# These are dimensionless and FIXED; only the measured bond length r0 sets the
# physical scale, through  sigma = r0 / SIGMA_RATIO.
ZHOU_A = 7.0496           # 2-body prefactor A
ZHOU_B = 1.116149         # 2-body coefficient B
SIGMA_RATIO = 1.28094     # r0 / sigma   (where the SW well sits)
CUTOFF_RATIO = 1.953387   # cutoff / sigma   (the SW parameter 'a')

BOND_MAX_ANG = 3.3        # a true first-neighbour bond is shorter than this (Angstrom)
INORGANIC = {"Cd", "Se"}  # the framework elements; C/H/O are ligand atoms

#  Loading
def read_trajectory(filepath, fmt="extxyz", first_n=None, skip_n=None):
    ase_format = "lammps-dump-text" if fmt == "lammps" else "extxyz"
    frames = ase.io.read(filepath, index=":", format=ase_format)
    if skip_n:
        frames = frames[skip_n:]
    if first_n:
        frames = frames[:first_n]

    symbols = frames[0].get_chemical_symbols()
    positions = []
    forces = []
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        frame_forces = frame.arrays.get("forces")
        if frame_forces is None and frame.calc is not None:
            frame_forces = frame.calc.results.get("forces")
        if frame_forces is None:
            forces.append(None)
        else:
            forces.append(frame_forces / FORCE_AU_TO_EV_ANG)

    print(f"  {len(frames)} frames, {len(symbols)} atoms")
    return symbols, positions, forces


def is_scoped_pair(element_a, element_b):
    cd_se_bond = element_a in INORGANIC and element_b in INORGANIC and element_a != element_b
    metal_oxygen_bond = ((element_a in INORGANIC and element_b == "O")
                         or (element_b in INORGANIC and element_a == "O"))
    return cd_se_bond or metal_oxygen_bond

#  Frozen 2-body scales, measured from the RDF
def bond_geometry_from_rdf(sampled_positions_bohr, atoms_a, atoms_b, same_element):
    bin_edges = np.linspace(1.8 * ANGSTROM_TO_BOHR, 5.5 * ANGSTROM_TO_BOHR, 75)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bond_max = BOND_MAX_ANG * ANGSTROM_TO_BOHR

    histogram = np.zeros(len(bin_centres))
    coordination = 0.0
    for positions in sampled_positions_bohr:
        distances = cdist(positions[atoms_a], positions[atoms_b])
        if same_element:
            np.fill_diagonal(distances, np.inf)        # don't count an atom with itself
        histogram += np.histogram(distances, bins=bin_edges)[0]
        coordination += (distances < bond_max).sum() / len(atoms_a)

    g = histogram / np.maximum(bin_centres**2, 1e-12)  # divide out the ~r^2 shell volume
    g = np.convolve(g, np.ones(3) / 3, mode="same")    # light 3-bin smoothing

    first_shell = bin_centres < bond_max
    if first_shell.sum() < 2:
        return None
    bond_length = float(bin_centres[first_shell][np.argmax(g[first_shell])])

    after_peak = (bin_centres > bond_length) & (bin_centres < bond_length + 2.0 * ANGSTROM_TO_BOHR)
    if after_peak.sum() < 2:
        return None
    first_minimum = float(bin_centres[after_peak][np.argmin(g[after_peak])])

    return bond_length, first_minimum, coordination / len(sampled_positions_bohr)


def measure_pair_scales(sampled_positions_bohr, symbols, elements):
    symbols = np.array(symbols)
    atoms_of_element = {element: np.where(symbols == element)[0] for element in elements}

    scales = {field: {} for field in ("r0", "sigma", "cutoff", "A", "B")}
    print("\nFrozen 2-body scales (Zhou II-VI shape; only r0 is measured):")
    for element_a, element_b in combinations_with_replacement(elements, 2):
        if not is_scoped_pair(element_a, element_b):
            continue
        atoms_a, atoms_b = atoms_of_element[element_a], atoms_of_element[element_b]
        if len(atoms_a) == 0 or len(atoms_b) == 0:
            continue

        geometry = bond_geometry_from_rdf(
            sampled_positions_bohr, atoms_a, atoms_b, same_element=(element_a == element_b))
        if geometry is None:
            continue
        bond_length, first_minimum, coordination = geometry
        bond = canonical_pair(element_a, element_b)
        if coordination < 0.3 or not (first_minimum > bond_length > 0):
            print(f"  {bond:>10}: not a bond (coordination={coordination:.2f}) -- skipped")
            continue

        sigma = bond_length / SIGMA_RATIO
        scales["r0"][bond] = bond_length
        scales["sigma"][bond] = sigma
        scales["cutoff"][bond] = CUTOFF_RATIO * sigma
        scales["A"][bond] = ZHOU_A
        scales["B"][bond] = ZHOU_B
        print(f"  {bond:>10}: r0={bond_length * BOHR_TO_ANGSTROM:.3f}  "
              f"sigma={sigma * BOHR_TO_ANGSTROM:.3f}  "
              f"cutoff={scales['cutoff'][bond] * BOHR_TO_ANGSTROM:.3f} Angstrom")
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

#  Per-frame graph: bonds and triplets
def build_edges(positions_bohr, symbols, cutoffs_bohr):
    edges = {}
    if not cutoffs_bohr:
        return edges

    positions_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM
    search_radius = max(cutoffs_bohr.values()) * BOHR_TO_ANGSTROM
    atoms = Atoms(symbols=list(symbols), positions=positions_ang)
    first, second, distance_ang = neighbor_list("ijd", atoms, search_radius, self_interaction=False)

    symbols = np.array(symbols)
    once = first < second                              # neighbor_list lists each pair twice
    for atom_i, atom_j, distance_bohr in zip(first[once], second[once],
                                             distance_ang[once] / BOHR_TO_ANGSTROM):
        bond = canonical_pair(symbols[atom_i], symbols[atom_j])
        if bond in cutoffs_bohr and distance_bohr <= cutoffs_bohr[bond]:
            i_list, j_list = edges.setdefault(bond, ([], []))
            i_list.append(atom_i)
            j_list.append(atom_j)

    return {bond: (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long))
            for bond, (i, j) in edges.items()}


def build_triplets(edges, symbols, triplet_names):
    # collect each atom's neighbours from the (one-directional) bond lists
    neighbours = {}
    for atom_i_tensor, atom_j_tensor in edges.values():
        for atom_i, atom_j in zip(atom_i_tensor.tolist(), atom_j_tensor.tolist()):
            neighbours.setdefault(atom_i, []).append(atom_j)
            neighbours.setdefault(atom_j, []).append(atom_i)

    symbols = np.array(symbols)
    triplet_names = set(triplet_names)
    triplets = {}
    for centre, centre_neighbours in neighbours.items():
        for atom_a, atom_b in combinations(centre_neighbours, 2):
            name = canonical_triplet(symbols[centre], symbols[atom_a], symbols[atom_b])
            if name not in triplet_names:
                continue
            if symbols[atom_b] < symbols[atom_a]:      # order legs to match the name
                atom_a, atom_b = atom_b, atom_a
            centre_list, a_list, b_list = triplets.setdefault(name, ([], [], []))
            centre_list.append(centre)
            a_list.append(atom_a)
            b_list.append(atom_b)

    return {name: (torch.tensor(c, dtype=torch.long), torch.tensor(a, dtype=torch.long),
                   torch.tensor(b, dtype=torch.long))
            for name, (c, a, b) in triplets.items()}


def make_batch(frames, atoms_per_frame):
    positions = torch.cat([frame["positions"] for frame in frames])
    dft_forces = torch.cat([frame["dft_forces"] for frame in frames])
    fit_mask = torch.cat([frame["fit_mask"] for frame in frames])

    edges = {}
    triplets = {}
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
    triplets = {name: (torch.cat(c), torch.cat(a), torch.cat(b))
                for name, (c, a, b) in triplets.items()}
    return {"positions": positions, "dft_forces": dft_forces, "fit_mask": fit_mask,
            "edges": edges, "triplets": triplets}

#  Dataset: load everything and build one graph per frame
class DFTDataset:
    def __init__(self, dataset_configs, first_n_frames=None):
        print(f"\n{'=' * 60}\nLOADING DATA\n{'=' * 60}")
        self.symbols = None
        self.positions = []       # list of (n_atoms, 3) Bohr arrays
        self.forces = []          # list of (n_atoms, 3) au arrays (or None)
        for dataset in dataset_configs:
            print(f"\n{dataset.get('name', dataset['xyz'])}")
            symbols, positions, forces = read_trajectory(
                dataset["xyz"], fmt=dataset.get("format", "extxyz"),
                first_n=dataset.get("first_n_frames", first_n_frames),
                skip_n=dataset.get("skip_frames"))
            self.symbols = symbols
            self.positions += positions
            self.forces += forces

        self.atoms_per_frame = len(self.symbols)
        self.elements = sorted(set(self.symbols))
        self.fit_mask = np.array([element in INORGANIC for element in self.symbols])

        counts = Counter(self.symbols)
        print("\nElements (" + ", ".join(f"{e}={counts[e]}" for e in self.elements) + ")")
        print(f"{len(self.positions)} frames, {self.atoms_per_frame} atoms each")

    def build_graphs(self):
        sample = self.positions[:: max(1, len(self.positions) // 50)]
        sampled = [np.asarray(p) for p in sample]
        self.scales = measure_pair_scales(sampled, self.symbols, self.elements)
        self.cutoffs_bohr = self.scales["cutoff"]
        self.triplet_type_names = enumerate_triplet_types(self.elements, self.cutoffs_bohr)
        print(f"Scoped bonds: {len(self.cutoffs_bohr)}, "
              f"triplet types: {len(self.triplet_type_names)}")

        fit_mask = torch.tensor(self.fit_mask, dtype=torch.bool)
        print("\nBuilding graphs...")
        self.graphs = []
        for positions_bohr, forces_au in zip(self.positions, self.forces):
            edges = build_edges(positions_bohr, self.symbols, self.cutoffs_bohr)
            if not edges:
                continue
            triplets = build_triplets(edges, self.symbols, self.triplet_type_names)
            positions = torch.tensor(np.asarray(positions_bohr), dtype=torch.float64)
            dft_forces = (torch.tensor(forces_au, dtype=torch.float64)
                          if forces_au is not None else torch.zeros_like(positions))
            self.graphs.append({"positions": positions, "dft_forces": dft_forces,
                                "fit_mask": fit_mask, "edges": edges, "triplets": triplets})
        print(f"done ({len(self.graphs)} graphs)")
