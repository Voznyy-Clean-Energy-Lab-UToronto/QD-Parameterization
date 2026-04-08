import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin
from torch_geometric.data import Data

from .utils import (
    ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG,
    canonical_pair,
)


def read_extxyz(filepath, first_n=None):
    frames = ase.io.read(filepath, index=':', format='extxyz')
    symbols = frames[0].get_chemical_symbols()
    print(f"  Frames: {len(frames)}, Atoms: {len(symbols)}")

    if first_n and first_n < len(frames):
        frames = frames[:first_n]
        print(f"  Using first {first_n} frames")

    positions, forces = [], []
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        try:
            forces.append(frame.get_forces() / FORCE_AU_TO_EV_ANG)
        except Exception:
            forces.append(None)

    return symbols, positions, forces


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

    return atom_symbols, positions_list, forces_list


#  RDF-based cutoff detection

def compute_rdf_cutoff(frame_data, symbols, elem_a, elem_b,
                       r_max_ang=8.0, bin_width_ang=0.05,
                       sigma=2.0, stride=10, min_peak_height=0.1,
                       threshold_frac=0.02):
#Find first shell cutoff using RDF. After first peak, find minima. Minima threshold is 2% of max peak height.
    symbols_arr = np.array(symbols)
    idx_a = np.where(symbols_arr == elem_a)[0]
    idx_b = np.where(symbols_arr == elem_b)[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        return None

    same_species = (elem_a == elem_b)
    n_bins = int(r_max_ang / bin_width_ang)
    r_edges_ang = np.linspace(0, r_max_ang, n_bins + 1)
    r_centers_ang = 0.5 * (r_edges_ang[:-1] + r_edges_ang[1:])
    hist = np.zeros(n_bins, dtype=np.float64)
    n_frames_used = 0

    for fi in range(0, len(frame_data), max(1, stride)):
        _, pos_bohr = frame_data[fi]
        pos_ang = np.asarray(pos_bohr) * BOHR_TO_ANGSTROM

        dists = cdist(pos_ang[idx_a], pos_ang[idx_b]).ravel()
        if same_species:
            dists = dists[dists > 0.01]

        hist += np.histogram(dists, bins=r_edges_ang)[0]
        n_frames_used += 1

    if n_frames_used == 0 or hist.sum() == 0:
        return None

    # Smooth
    smoothed = gaussian_filter1d(hist.astype(np.float64), sigma=sigma)

    # Normalize so max = 1
    max_val = smoothed.max()
    if max_val <= 0:
        return None
    smoothed_norm = smoothed / max_val

    # Find peaks
    peak_idx = argrelmax(smoothed_norm, order=3)[0]
    peak_idx = peak_idx[smoothed_norm[peak_idx] > min_peak_height]

    if len(peak_idx) == 0:
        return None

    first_peak = peak_idx[0]
    peak_height = smoothed_norm[first_peak]

    #find where the RDF drops below a threshold after the first peak.
    drop_threshold = peak_height * threshold_frac

    #find where smoothed drops below threshold after peak
    after_peak = smoothed_norm[first_peak:]
    below_thresh = np.where(after_peak < drop_threshold)[0]

    if len(below_thresh) > 0:
        cutoff_idx = first_peak + below_thresh[0]
        cutoff_ang = r_centers_ang[cutoff_idx]
    else:
        #look for formal minimum (argrelmin)
        min_idx = argrelmin(smoothed_norm, order=3)[0]
        min_after_peak = min_idx[min_idx > first_peak]
        if len(min_after_peak) == 0:
            min_idx_wide = argrelmin(smoothed_norm, order=5)[0]
            min_after_peak = min_idx_wide[min_idx_wide > first_peak]
        if len(min_after_peak) == 0:
            return None
        cutoff_idx = min_after_peak[0]
        cutoff_ang = r_centers_ang[cutoff_idx]

    peak_ang = r_centers_ang[first_peak]
    if cutoff_ang <= peak_ang or cutoff_ang > r_max_ang * 0.95:
        return None

    return cutoff_ang * ANGSTROM_TO_BOHR

#The dreaded "gap detection". In order to find if it is "first shell" or longer, bypassing knn_edges (pseudo-coordination number), it detects if there's a "gap"
# For example, if I have a max CN of 4, and 4 nearest neighbours are at: 2.30A, 2.32A, 2.34A, 5.32A, then I know the 5.32A is in a different "shell" and wont be counted.
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

    return median


#  Edge building

def build_knn_edges(positions_bohr, atom_symbols, knn_config,
                    precomputed_cutoffs=None):
#Build edges for 1 frame, include all pairs within cutoff using ASE neighbor_list.
    inorganic = set(knn_config['inorganic_elements'])
    max_neighbors = knn_config.get('max_gap_neighbors', 20)
    ligand_cutoff_bohr = knn_config.get('ligand_cutoff', 4.0) * ANGSTROM_TO_BOHR

    pos = np.asarray(positions_bohr)
    pos_ang = pos * BOHR_TO_ANGSTROM
    symbols = np.array(atom_symbols)

    inorganic_atom_indices = {}
    for elem in sorted(inorganic):
        mask = symbols == elem
        if mask.any():
            inorganic_atom_indices[elem] = np.where(mask)[0]
    ligand_idx = np.where(~np.isin(symbols, list(inorganic)))[0]
    inorg_elems = sorted(inorganic_atom_indices.keys())

    edge_arrays = []
    pair_cutoffs_bohr = {}

    # Determine per-pair cutoffs (precomputed or gap detection)
    for i, elem_a in enumerate(inorg_elems):
        for elem_b in inorg_elems[i:]:
            pair_key = canonical_pair(elem_a, elem_b)
            if precomputed_cutoffs is not None and pair_key in precomputed_cutoffs:
                pair_cutoffs_bohr[pair_key] = precomputed_cutoffs[pair_key]
            else:
                # Need distance matrix for gap detection (first-frame only)
                idx_a = inorganic_atom_indices[elem_a]
                idx_b = inorganic_atom_indices[elem_b]
                if elem_a == elem_b:
                    dists = cdist(pos[idx_a], pos[idx_a])
                    np.fill_diagonal(dists, np.inf)
                else:
                    dists = cdist(pos[idx_a], pos[idx_b])
                pair_cutoffs_bohr[pair_key] = detect_gap_cutoff(dists, max_neighbors)

    # Ligand cutoffs
    if len(ligand_idx) > 0:
        ligand_elems = sorted(set(symbols[ligand_idx]))
        for ie in inorg_elems:
            for le in ligand_elems:
                pair_cutoffs_bohr[canonical_pair(ie, le)] = ligand_cutoff_bohr

    # Determine max cutoff across all pairs for ASE neighbor list
    max_cutoff_ang = max(c * BOHR_TO_ANGSTROM for c in pair_cutoffs_bohr.values()) if pair_cutoffs_bohr else 7.5

    # Build ASE neighbor list (non-periodic)
    atoms = Atoms(symbols=list(atom_symbols), positions=pos_ang)
    i_list, j_list, d_list = neighbor_list('ijd', atoms, max_cutoff_ang,
                                            self_interaction=False)

    # Convert distances back to Bohr for cutoff comparison
    d_bohr = d_list / BOHR_TO_ANGSTROM

    # Filter edges by per-pair cutoff
    for edge_idx in range(len(i_list)):
        ai, aj = int(i_list[edge_idx]), int(j_list[edge_idx])
        if ai >= aj:
            continue  # only keep i < j to avoid duplicates
        ei, ej = symbols[ai], symbols[aj]
        pair_key = canonical_pair(ei, ej)
        cutoff = pair_cutoffs_bohr.get(pair_key)
        if cutoff is None:
            continue
        # Check if this pair type is relevant (inorganic-inorganic or inorganic-ligand)
        ei_inorg = ei in inorganic
        ej_inorg = ej in inorganic
        if not ei_inorg and not ej_inorg:
            continue  # skip ligand-ligand
        if d_bohr[edge_idx] <= cutoff:
            edge_arrays.append(np.array([[ai, aj]]))

    # Deduplicate and symmetrize to directed edges
    if edge_arrays:
        unique_edges = np.unique(np.vstack(edge_arrays), axis=0)
    else:
        unique_edges = np.empty((0, 2), dtype=np.int64)

    src = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]]).astype(np.int64)
    tgt = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]]).astype(np.int64)
    pair_cutoffs_angstrom = {pair: cut * BOHR_TO_ANGSTROM
                             for pair, cut in pair_cutoffs_bohr.items()}

    return src, tgt, pair_cutoffs_angstrom


#  DFTDataset

class DFTDataset:

    def __init__(self, dataset_configs, knn_config, first_n_frames=None):
        self.frame_data = []      # [(symbols, pos_bohr), ...]
        self.forces_data = []     # [forces_au or None, ...]
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
                symbols, pos_list, frc_list = read_lammps_dump(
                    ds['xyz'], first_n=ds_first_n)
            else:
                symbols, pos_list, frc_list = read_extxyz(
                    ds['xyz'], first_n=ds_first_n)

            all_elements.update(symbols)
            self.frame_data.extend([(symbols, p) for p in pos_list])
            self.forces_data.extend(frc_list)

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

        print(f"\nTotal: {len(self.frame_data)} frames, Elements: {self.elements}")
        print(f"Pairs ({len(self.pair_names)}): {', '.join(self.pair_names)}")

    def _compute_cutoffs(self):
    #Compute cutoffs for edge building. Inorganic (Cd, Se) use the fist shell cutoff found, ligands use whatever is defined in config (4.0A by default)
        inorganic = set(self.knn_config['inorganic_elements'])
        symbols = self.frame_data[0][0]
        symbols_arr = np.array(symbols)
        positions_bohr = self.frame_data[0][1]
        pos = np.asarray(positions_bohr)
        inorg_elems = sorted(e for e in set(symbols) if e in inorganic)
        ligand_cutoff_ang = self.knn_config.get('ligand_cutoff', 4.0)
        ligand_cutoff_bohr = ligand_cutoff_ang * ANGSTROM_TO_BOHR
        max_neighbors = self.knn_config.get('max_gap_neighbors', 20)

        print(f"\nComputing first-shell gap cutoffs...")

        cutoffs_bohr = {}

        # Inorganic pairs: gap cutoff from first frame distance matrix
        # (detect_gap_cutoff requires full distance matrix, so cdist stays here)
        inorganic_atom_indices = {}
        for elem in inorg_elems:
            mask = symbols_arr == elem
            if mask.any():
                inorganic_atom_indices[elem] = np.where(mask)[0]

        for i, elem_a in enumerate(inorg_elems):
            for elem_b in inorg_elems[i:]:
                pair_key = canonical_pair(elem_a, elem_b)
                idx_a = inorganic_atom_indices[elem_a]
                idx_b = inorganic_atom_indices[elem_b]

                if elem_a == elem_b:
                    dists = cdist(pos[idx_a], pos[idx_a])
                    np.fill_diagonal(dists, np.inf)
                else:
                    dists = cdist(pos[idx_a], pos[idx_b])

                gap_cut_bohr = detect_gap_cutoff(dists, max_neighbors)
                gap_cut_ang = gap_cut_bohr * BOHR_TO_ANGSTROM

                cutoffs_bohr[pair_key] = gap_cut_bohr
                print(f"  {pair_key}: {gap_cut_ang:.2f} A (1st shell gap)")

        # Inorganic-ligand pairs: fixed cutoff
        ligand_idx = np.where(~np.isin(symbols_arr, list(inorganic)))[0]
        if len(ligand_idx) > 0:
            ligand_elems = sorted(set(symbols_arr[ligand_idx]))
            for ie in inorg_elems:
                for le in ligand_elems:
                    pk = canonical_pair(ie, le)
                    cutoffs_bohr[pk] = ligand_cutoff_bohr
                    print(f"  {pk}: {ligand_cutoff_ang:.1f} A (ligand)")

        self.cutoffs_bohr = cutoffs_bohr

    def build_graphs(self):
        # Compute stable cutoffs once across all frames
        self._compute_cutoffs()

        print("\nBuilding graphs...", end=" ", flush=True)
        self.graphs = []
        pair_lookup_t = torch.tensor(self.pair_lookup, dtype=torch.long)

        # gap_cuts_angstrom used by LAMMPS export for cutoffs
        self.gap_cuts_angstrom = {pair: cut * BOHR_TO_ANGSTROM
                                  for pair, cut in self.cutoffs_bohr.items()}

        for i, (symbols, positions_bohr) in enumerate(self.frame_data):
            # Build edges using precomputed cutoffs
            src, tgt, _ = build_knn_edges(
                positions_bohr, symbols, self.knn_config,
                precomputed_cutoffs=self.cutoffs_bohr)

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

            self.graphs.append(Data(
                pos=pos,
                edge_index=torch.stack([src_t, tgt_t]),
                distances=dist,
                edge_unit_vectors=unit_vec,
                element_indices=elem_idx,
                pair_indices=pair_indices,
                dft_forces=dft_forces,
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
