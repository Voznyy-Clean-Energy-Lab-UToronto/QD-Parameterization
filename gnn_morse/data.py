import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from .utils import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG, canonical_pair


def read_trajectory(filepath, fmt='extxyz', first_n=None):
    ase_format = 'lammps-dump-text' if fmt == 'lammps' else 'extxyz'
    frames = ase.io.read(filepath, index=':', format=ase_format)
    symbols = frames[0].get_chemical_symbols()
    print(f"  Frames: {len(frames)}, Atoms: {len(symbols)}")

    if first_n and first_n < len(frames):
        frames = frames[:first_n]
        print(f"  Using first {first_n} frames")

    positions = []
    forces = []
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        f = frame.arrays.get('forces', frame.calc.results.get('forces') if frame.calc else None)
        if f is not None:
            forces.append(f / FORCE_AU_TO_EV_ANG)
        else:
            forces.append(None)

    return symbols, positions, forces


def detect_gap_cutoff(dist_matrix, max_neighbors=20):
    dists = dist_matrix.copy()
    dists[(dists < 1e-10) | ~np.isfinite(dists)] = np.inf

    if dists.size == 0 or np.all(np.isinf(dists)):
        return float('inf')

    sorted_dists = np.sort(dists, axis=1)
    max_nn = min(max_neighbors + 1, sorted_dists.shape[1])
    nearest = sorted_dists[:, :max_nn]

    has_neighbors = np.sum(np.isfinite(nearest), axis=1) >= 2
    if not has_neighbors.any():
        return float('inf')
    gaps = np.diff(nearest, axis=1)
    gaps[~np.isfinite(gaps)] = 0

    max_gap_pos = np.argmax(gaps, axis=1)
    rows = np.arange(len(nearest))
    cutoffs = 0.5 * (nearest[rows, max_gap_pos] + nearest[rows, max_gap_pos + 1])
    cutoffs[~has_neighbors | ~np.isfinite(cutoffs)] = np.inf

    finite = cutoffs[np.isfinite(cutoffs)]
    return float(np.median(finite)) if len(finite) > 0 else float('inf')


def build_edges(positions_bohr, symbols, knn_config, cutoffs_bohr):
    inorganic = set(knn_config['inorganic_elements'])
    symbols_arr = np.array(symbols)
    pos_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM

    max_cut_ang = max(c * BOHR_TO_ANGSTROM for c in cutoffs_bohr.values())
    atoms = Atoms(symbols=list(symbols), positions=pos_ang)
    i_arr, j_arr, d_arr = neighbor_list('ijd', atoms, max_cut_ang, self_interaction=False)
    d_bohr = d_arr / BOHR_TO_ANGSTROM

    # keep i < j to avoid duplicates, at least one must be inorganic
    keep = i_arr < j_arr
    ai, aj, d = i_arr[keep], j_arr[keep], d_bohr[keep]
    ei, ej = symbols_arr[ai], symbols_arr[aj]
    has_inorg = np.isin(ei, sorted(inorganic)) | np.isin(ej, sorted(inorganic))
    ai, aj, d = ai[has_inorg], aj[has_inorg], d[has_inorg]
    ei, ej = ei[has_inorg], ej[has_inorg]

    # filter by per-pair cutoff
    pair_keys = np.where(ei <= ej,
                         np.char.add(np.char.add(ei, '-'), ej),
                         np.char.add(np.char.add(ej, '-'), ei))
    cutoff_map = {pk: cutoffs_bohr.get(pk, np.nan) for pk in np.unique(pair_keys)}
    cutoffs = np.array([cutoff_map[pk] for pk in pair_keys])
    valid = np.isfinite(cutoffs) & (d <= cutoffs)

    edges = np.column_stack([ai[valid], aj[valid]]) if valid.any() else np.empty((0, 2), dtype=np.int64)
    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    tgt = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    return src, tgt


class DFTDataset:
    def __init__(self, dataset_configs, knn_config, first_n_frames=None):
        self.frame_data = []
        self.forces_data = []

        print(f"\n{'='*60}\nLOADING DATA\n{'='*60}")

        for ds in dataset_configs:
            name = ds.get('name', ds['xyz'])
            fmt = ds.get('format', 'extxyz')
            n_frames = ds.get('first_n_frames', first_n_frames)
            print(f"\n{name} (format: {fmt})")

            symbols, pos_list, frc_list = read_trajectory(ds['xyz'], fmt=fmt, first_n=n_frames)
            self.frame_data.extend([(symbols, p) for p in pos_list])
            self.forces_data.extend(frc_list)

        self.elements = sorted(set(self.frame_data[0][0]))
        self.element_to_index = {e: i for i, e in enumerate(self.elements)}
        self.knn_config = knn_config

        # element pair names and lookup
        ne = len(self.elements)
        self.pair_names = [f"{e1}-{e2}" for e1, e2 in combinations_with_replacement(self.elements, 2)]
        self.pair_name_to_index = {pn: i for i, pn in enumerate(self.pair_names)}
        self.pair_lookup = np.zeros((ne, ne), dtype=np.int64)
        for i in range(ne):
            for j in range(ne):
                lo, hi = min(i, j), max(i, j)
                self.pair_lookup[i, j] = self.pair_name_to_index[f"{self.elements[lo]}-{self.elements[hi]}"]

        print(f"\nTotal: {len(self.frame_data)} frames, Elements: {self.elements}")

    def build_graphs(self):
        #detect per-pair cutoffs from frame 0, then buildgraphs
        inorganic = set(self.knn_config['inorganic_elements'])
        symbols = self.frame_data[0][0]
        symbols_arr = np.array(symbols)
        pos = np.asarray(self.frame_data[0][1])
        max_neighbors = self.knn_config.get('max_gap_neighbors', 20)
        ligand_cutoff = self.knn_config.get('ligand_cutoff', 4.0) * ANGSTROM_TO_BOHR
        max_lig_cut = self.knn_config.get('max_ligand_cutoff', 6.0) * ANGSTROM_TO_BOHR

        self.cutoffs_bohr = {}

        # inorganic-inorganic cutoffs
        inorg_elems = sorted(e for e in set(symbols) if e in inorganic)
        inorg_indices = {e: np.where(symbols_arr == e)[0] for e in inorg_elems}

        print(f"\nCutoffs (gap-detected):")
        for i, ea in enumerate(inorg_elems):
            for eb in inorg_elems[i:]:
                pk = canonical_pair(ea, eb)
                if ea == eb:
                    d = cdist(pos[inorg_indices[ea]], pos[inorg_indices[ea]])
                    np.fill_diagonal(d, np.inf)
                else:
                    d = cdist(pos[inorg_indices[ea]], pos[inorg_indices[eb]])
                cut = detect_gap_cutoff(d, max_neighbors)
                self.cutoffs_bohr[pk] = cut
                print(f"  {pk}: {cut * BOHR_TO_ANGSTROM:.2f} A")

        # inorganic-ligand cutoffs
        ligand_idx = np.where(~np.isin(symbols_arr, list(inorganic)))[0]
        if len(ligand_idx) > 0:
            for ie in inorg_elems:
                for le in sorted(set(symbols_arr[ligand_idx])):
                    pk = canonical_pair(ie, le)
                    d = cdist(pos[inorg_indices[ie]], pos[np.where(symbols_arr == le)[0]])
                    cut = detect_gap_cutoff(d, max_neighbors)
                    cut = min(cut, max_lig_cut) if np.isfinite(cut) else ligand_cutoff
                    self.cutoffs_bohr[pk] = cut
                    print(f"  {pk}: {cut * BOHR_TO_ANGSTROM:.2f} A")

        self.gap_cuts_angstrom = {p: c * BOHR_TO_ANGSTROM for p, c in self.cutoffs_bohr.items()}

        # build graphs
        print("\nBuilding graphs...", end=" ", flush=True)
        self.graphs = []
        pair_lookup_t = torch.tensor(self.pair_lookup, dtype=torch.long)

        for i, (syms, pos_bohr) in enumerate(self.frame_data):
            src, tgt = build_edges(pos_bohr, syms, self.knn_config, self.cutoffs_bohr)
            if len(src) == 0:
                continue

            pos_t = torch.tensor(pos_bohr, dtype=torch.float64)
            src_t = torch.tensor(src, dtype=torch.long)
            tgt_t = torch.tensor(tgt, dtype=torch.long)

            disp = pos_t[tgt_t] - pos_t[src_t]
            dist = torch.norm(disp, dim=1)
            unit_vec = disp / dist.clamp(min=1e-10).unsqueeze(1)

            elem_idx = torch.tensor([self.element_to_index[s] for s in syms], dtype=torch.long)
            pair_indices = pair_lookup_t[elem_idx[src_t], elem_idx[tgt_t]]

            frc = self.forces_data[i]
            dft_forces = torch.tensor(frc, dtype=torch.float64) if frc is not None else torch.zeros_like(pos_t)

            self.graphs.append(Data(
                pos=pos_t, edge_index=torch.stack([src_t, tgt_t]),
                distances=dist, edge_unit_vectors=unit_vec,
                element_indices=elem_idx, pair_indices=pair_indices,
                dft_forces=dft_forces,
            ))

        print(f"done ({len(self.graphs)} graphs)")
