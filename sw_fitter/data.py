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

TAIL_FACTOR = 1.20
BOND_MAX_ANG = 7.0

def read_trajectory(filepath, first_n=None, skip_n=None):
    frames = ase.io.read(filepath, index=":", format="extxyz")
    if skip_n:
        frames = frames[skip_n:]
    if first_n:
        frames = frames[:first_n]

    symbols = frames[0].get_chemical_symbols()
    positions = [frame.positions * ANGSTROM_TO_BOHR for frame in frames]
    forces = [frame.get_forces() / FORCE_AU_TO_EV_ANG for frame in frames]

    print(f"  {len(frames)} frames, {len(symbols)} atoms")
    return symbols, positions, forces


def is_candidate_pair(element_a, element_b, fit_elements, ligand_elements):
    scope = fit_elements | ligand_elements
    if element_a not in scope or element_b not in scope:
        return False
    return element_a in fit_elements or element_b in fit_elements


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


    after_peak = (bin_centres > r0_bohr) & (bin_centres < r0_bohr + 2.0 * ANGSTROM_TO_BOHR)
    if after_peak.sum() < 2:
        return None
    first_min_bohr = float(bin_centres[after_peak][np.argmin(g[after_peak])])

    coordination = coordination / len(sampled_positions_bohr)

    return r0_bohr, first_min_bohr, coordination


def measure_pair_scales(sampled_positions_bohr, symbols, elements, fit_elements, ligand_elements):
    symbols = np.array(symbols)
    atoms_of_element = {e: np.where(symbols == e)[0] for e in elements}

    scales = {field: {} for field in ("r0", "cutoff")}

    print("\n2-body scales (r0 and first-shell minimum measured from RDF):")
    for element_a, element_b in combinations_with_replacement(elements, 2):
        if not is_candidate_pair(element_a, element_b, fit_elements, ligand_elements):
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
        r0_bohr, first_min_bohr, coordination = geometry
        bond = canonical_pair(element_a, element_b)

        if coordination < 0.05 or not (first_min_bohr > r0_bohr > 0):
            print(f"  {bond:>8}: no clear bond (coordination={coordination:.2f}) — skipped")
            continue

        scales["r0"][bond] = r0_bohr
        scales["cutoff"][bond] = TAIL_FACTOR * first_min_bohr

        print(
            f"  {bond:>8}: r0={r0_bohr * BOHR_TO_ANGSTROM:.3f} Å  "
            f"cutoff={scales['cutoff'][bond] * BOHR_TO_ANGSTROM:.3f} Å"
        )
    return scales


def enumerate_triplet_types(elements, scoped_bonds, fit_elements):
    names = set()
    for centre in elements:
        if centre not in fit_elements:
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


def compute_geometry(positions, edges, triplets):
    edge_len = {}
    for bond, (atom_i, atom_j) in edges.items():
        edge_len[bond] = (positions[atom_j] - positions[atom_i]).norm(dim=1)
    tri_len = {}
    for name, (centre, atom_a, atom_b) in triplets.items():
        vec_ca = positions[atom_a] - positions[centre]
        vec_cb = positions[atom_b] - positions[centre]
        length_ca = vec_ca.norm(dim=1)
        length_cb = vec_cb.norm(dim=1)
        cos_theta = (vec_ca * vec_cb).sum(dim=1) / (length_ca * length_cb)
        tri_len[name] = (length_ca, length_cb, cos_theta)
    return edge_len, tri_len


def make_batch(frames):
    positions = torch.cat([f["positions"] for f in frames])
    dft_forces = torch.cat([f["dft_forces"] for f in frames])
    fit_mask = torch.cat([f["fit_mask"] for f in frames])

    edges, triplets, edge_len, tri_len = {}, {}, {}, {}
    offset = 0
    for frame in frames:
        for bond, (atom_i, atom_j) in frame["edges"].items():
            i_list, j_list = edges.setdefault(bond, ([], []))
            i_list.append(atom_i + offset)
            j_list.append(atom_j + offset)
            edge_len.setdefault(bond, []).append(frame["edge_len"][bond])
        for name, (centre, atom_a, atom_b) in frame["triplets"].items():
            c_list, a_list, b_list = triplets.setdefault(name, ([], [], []))
            c_list.append(centre + offset)
            a_list.append(atom_a + offset)
            b_list.append(atom_b + offset)
            tri_len.setdefault(name, []).append(frame["tri_len"][name])
        offset += frame["positions"].shape[0]

    edges = {bond: (torch.cat(i), torch.cat(j)) for bond, (i, j) in edges.items()}
    triplets = {
        name: (torch.cat(c), torch.cat(a), torch.cat(b))
        for name, (c, a, b) in triplets.items()
    }
    edge_len = {bond: torch.cat(parts) for bond, parts in edge_len.items()}
    tri_len = {
        name: (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]))
        for name, parts in tri_len.items()
    }
    return {
        "positions": positions,
        "dft_forces": dft_forces,
        "fit_mask": fit_mask,
        "edges": edges,
        "triplets": triplets,
        "edge_len": edge_len,
        "tri_len": tri_len,
    }


class DFTDataset:
    def __init__(self, dataset_configs, scope, first_n_frames=None):
        print(f"\n{'=' * 60}\nLOADING DATA\n{'=' * 60}")

        self.fit_elements = set(scope["fit_elements"])
        self.ligand_elements = set(scope.get("ligand_bond_elements", []))

        self._datasets = []
        for cfg in dataset_configs:
            print(f"\n{cfg.get('name', cfg['xyz'])}")
            symbols, positions, forces = read_trajectory(
                cfg["xyz"],
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

        framework_elems = sorted(e for e in self.elements if e in self.fit_elements)
        self.chemical_formula = "".join(framework_elems) + "_universal"

        total_frames = sum(len(ds["positions"]) for ds in self._datasets)
        print(f"\nElements: {self.elements}")
        print(f"{len(self._datasets)} dataset(s), {total_frames} frames total")

    def build_graphs(self):

        ref = self._datasets[0]
        ref_sample = ref["positions"][:: max(1, len(ref["positions"]) // 50)]
        self.scales = measure_pair_scales(
            [np.asarray(p) for p in ref_sample],
            ref["symbols"],
            sorted(set(ref["symbols"])),
            self.fit_elements,
            self.ligand_elements,
        )
        self.cutoffs_bohr = self.scales["cutoff"]
        self.triplet_type_names = enumerate_triplet_types(self.elements, self.cutoffs_bohr, self.fit_elements)
        print(
            f"Scoped bonds: {len(self.cutoffs_bohr)}, "
            f"triplet types: {len(self.triplet_type_names)}"
        )

        print("\nBuilding graphs at the fixed SW cutoff "
              f"(TAIL_FACTOR={TAIL_FACTOR} x RDF first minimum; graph radius = SW cutoff):")
        for bond in sorted(self.cutoffs_bohr):
            print(f"  {bond:>8}: graph radius {self.cutoffs_bohr[bond] * BOHR_TO_ANGSTROM:.2f} Å")
        self.graphs = []
        self.graphs_per_dataset = []
        for ds in self._datasets:
            symbols   = ds["symbols"]
            fit_mask  = torch.tensor(
                [e in self.fit_elements for e in symbols], dtype=torch.bool
            )
            qd_graphs = []
            for positions_bohr, forces_au in zip(ds["positions"], ds["forces"]):
                edges = build_edges(positions_bohr, symbols, self.cutoffs_bohr)
                if not edges:
                    continue
                triplets = build_triplets(edges, symbols, self.triplet_type_names)
                positions_t = torch.tensor(np.asarray(positions_bohr), dtype=torch.float64)
                dft_forces = torch.tensor(np.asarray(forces_au), dtype=torch.float64)
                edge_len, tri_len = compute_geometry(positions_t, edges, triplets)
                graph = {
                    "positions":  positions_t,
                    "dft_forces": dft_forces,
                    "fit_mask":   fit_mask,
                    "edges":      edges,
                    "triplets":   triplets,
                    "edge_len":   edge_len,
                    "tri_len":    tri_len,
                }
                self.graphs.append(graph)
                qd_graphs.append(graph)
            self.graphs_per_dataset.append(qd_graphs)
        print(f"done ({len(self.graphs)} graphs)")
