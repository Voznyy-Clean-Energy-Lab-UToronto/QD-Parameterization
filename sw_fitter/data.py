"""Data pipeline: trajectory loading, CN subtyping, edge/triplet building."""
import time
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from .utils import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG, canonical_pair


def read_trajectory(filepath, fmt='extxyz', first_n=None, skip_n=None):
    ase_format = 'lammps-dump-text' if fmt == 'lammps' else 'extxyz'
    frames = ase.io.read(filepath, index=':', format=ase_format)
    symbols = frames[0].get_chemical_symbols()
    print(f"  Frames: {len(frames)}, Atoms: {len(symbols)}")
    if skip_n and skip_n < len(frames):
        frames = frames[skip_n:]
    if first_n and first_n < len(frames):
        frames = frames[:first_n]
    print(f"  Using {len(frames)} frames")
    positions, forces = [], []
    # Extract cell from first frame (for PBC systems)
    cell_bohr = None
    pbc = frames[0].pbc
    if any(pbc):
        cell_bohr = frames[0].cell[:] * ANGSTROM_TO_BOHR
        print(f"  PBC detected: cell = {frames[0].cell.diagonal()} A")
    for frame in frames:
        positions.append(frame.positions * ANGSTROM_TO_BOHR)
        f = frame.arrays.get('forces',
                             frame.calc.results.get('forces') if frame.calc else None)
        forces.append(f / FORCE_AU_TO_EV_ANG if f is not None else None)
    return symbols, positions, forces, cell_bohr


def canonical_triplet(center, n1, n2):
    a, b = (n1, n2) if n1 <= n2 else (n2, n1)
    return f"{center}:{a}-{b}"


def solve_sw_shape(r0, r_cut, S_target, p=4.0):
    """SW (q=0) 2-body shape (sigma, a, B, A) such that the minimum sits at r0, the
    cutoff at r_cut, and the dimensionless curvature/depth ratio (-g''/g)*x_min^2
    equals S_target. x_min = r0/sigma. A normalises the well depth to eps.
    All lengths in the same units as r0,r_cut. Returns a dict."""
    def shape(x):                                  # x = x_min = r0/sigma
        a = r_cut * x / r0                         # a = r_cut/sigma
        c = 1.0 / (x - a) ** 2
        B = c * x ** (p + 1) / (p + c * x)         # from g'(x_min) = 0
        gfun = lambda z: (B * z ** -p - 1.0) * np.exp(1.0 / (z - a))
        h = 1e-4
        g0 = gfun(x)
        gpp = (gfun(x + h) - 2.0 * g0 + gfun(x - h)) / h ** 2
        return a, B, g0, (-gpp / g0) * x ** 2
    xs = np.linspace(1.02, 1.6, 140)
    S = np.array([shape(x)[3] for x in xs])
    x_min = float(xs[int(np.argmin(np.abs(S - S_target)))])
    a, B, g0, _ = shape(x_min)
    sigma = r0 / x_min
    return dict(sigma=sigma, a=a, B=float(B), A=float(-1.0 / g0),
                cutoff=a * sigma, x_min=x_min)


def build_edges(positions_bohr, raw_symbols, subtypes, cutoffs_bohr, cell_bohr=None):
    """Build bidirectional edges for ALL pairs within per-subtype-pair cutoffs."""
    sub_arr = np.array(subtypes)
    pos_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM
    if not cutoffs_bohr:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    max_cut_ang = max(c * BOHR_TO_ANGSTROM for c in cutoffs_bohr.values())
    if cell_bohr is not None:
        cell_ang = np.asarray(cell_bohr) * BOHR_TO_ANGSTROM
        atoms = Atoms(symbols=list(raw_symbols), positions=pos_ang,
                      cell=cell_ang, pbc=True)
    else:
        atoms = Atoms(symbols=list(raw_symbols), positions=pos_ang)
    i_arr, j_arr, d_arr = neighbor_list('ijd', atoms, max_cut_ang, self_interaction=False)
    d_bohr = d_arr / BOHR_TO_ANGSTROM

    # Keep i < j, then make bidirectional
    keep = i_arr < j_arr
    ai, aj, d = i_arr[keep], j_arr[keep], d_bohr[keep]

    # Filter by per-subtype-pair cutoff
    si, sj = sub_arr[ai], sub_arr[aj]
    pair_keys = np.where(si <= sj,
                         np.char.add(np.char.add(si, '-'), sj),
                         np.char.add(np.char.add(sj, '-'), si))
    cutoff_map = {pk: cutoffs_bohr.get(pk, np.nan) for pk in np.unique(pair_keys)}
    cutoffs = np.array([cutoff_map[pk] for pk in pair_keys])
    valid = np.isfinite(cutoffs) & (d <= cutoffs)

    edges = np.column_stack([ai[valid], aj[valid]]) if valid.any() else np.empty((0, 2), dtype=np.int64)
    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    tgt = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    return src, tgt


def build_triplets(edge_index, element_indices, pair_lookup, triplet_type_to_index,
                   subtype_labels):
    """Build triplets for ALL atoms (no organic/inorganic distinction)."""
    src, tgt = edge_index
    n_edges = len(src)
    empty = torch.empty(0, dtype=torch.long)
    if n_edges == 0:
        return empty, empty, empty, empty, empty, empty

    sort_idx = torch.argsort(src)
    sorted_src = src[sort_idx]
    sorted_tgt = tgt[sort_idx]
    sorted_pi = pair_lookup[element_indices[sorted_src], element_indices[sorted_tgt]]

    n_atoms = element_indices.size(0)
    counts = torch.zeros(n_atoms, dtype=torch.long)
    counts.scatter_add_(0, sorted_src, torch.ones(n_edges, dtype=torch.long))
    offsets = torch.zeros(n_atoms + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    active = torch.where(counts >= 2)[0]  # ALL atoms with 2+ neighbors
    if len(active) == 0:
        return empty, empty, empty, empty, empty, empty

    active_counts = counts[active]
    active_offsets = offsets[active]

    all_center, all_a, all_b, all_pi_a, all_pi_b = [], [], [], [], []
    for nn in torch.unique(active_counts):
        if nn < 2:
            continue
        nn_val = nn.item()
        mask = active_counts == nn
        centers_nn = active[mask]
        offsets_nn = active_offsets[mask]
        n_centers = len(centers_nn)
        local_a, local_b = torch.triu_indices(nn_val, nn_val, offset=1)
        n_pairs = len(local_a)
        center_exp = centers_nn.repeat_interleave(n_pairs)
        offset_exp = offsets_nn.repeat_interleave(n_pairs)
        global_a = offset_exp + local_a.repeat(n_centers)
        global_b = offset_exp + local_b.repeat(n_centers)
        all_center.append(center_exp)
        all_a.append(sorted_tgt[global_a])
        all_b.append(sorted_tgt[global_b])
        all_pi_a.append(sorted_pi[global_a])
        all_pi_b.append(sorted_pi[global_b])

    if not all_center:
        return empty, empty, empty, empty, empty, empty

    triplet_a = torch.cat(all_a)
    triplet_center = torch.cat(all_center)
    triplet_b = torch.cat(all_b)
    pi_ca = torch.cat(all_pi_a)
    pi_cb = torch.cat(all_pi_b)

    # Map to global triplet type indices
    sub_arr = np.array(subtype_labels)
    c_types = sub_arr[triplet_center.numpy()]
    a_types = sub_arr[triplet_a.numpy()]
    b_types = sub_arr[triplet_b.numpy()]
    triplet_names = [canonical_triplet(c, a, b) for c, a, b in zip(c_types, a_types, b_types)]
    triplet_type_idx = torch.tensor(
        [triplet_type_to_index.get(t, 0) for t in triplet_names], dtype=torch.long)

    return triplet_a, triplet_center, triplet_b, pi_ca, pi_cb, triplet_type_idx


class SWData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('triplet_a', 'triplet_center', 'triplet_b'):
            return self.pos.size(0)
        if key in ('triplet_pi_ca', 'triplet_pi_cb', 'pair_indices', 'triplet_type_idx'):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def assign_subtypes(symbols, positions_ang, inorganic_elements,
                    ligand_cutoff=3.0, inorganic_cutoff=3.5):
    """CN-based subtype assignment. Organic atoms keep their element name."""
    elements = np.array(symbols)
    inorganic = set(inorganic_elements)
    n_atoms = len(elements)
    dmat = cdist(positions_ang, positions_ang)
    np.fill_diagonal(dmat, np.inf)

    o_mask = elements == 'O'
    is_slig = np.zeros(n_atoms, dtype=bool)
    if o_mask.any():
        for i in range(n_atoms):
            if elements[i] in inorganic and np.any(dmat[i, o_mask] < ligand_cutoff):
                is_slig[i] = True

    inorg_mask = np.array([e in inorganic for e in elements])
    cn = np.zeros(n_atoms, dtype=int)
    for i in np.where(inorg_mask)[0]:
        cn[i] = np.sum(dmat[i, inorg_mask] < inorganic_cutoff)

    labels = list(symbols)
    for i in range(n_atoms):
        if elements[i] not in inorganic:
            continue
        if is_slig[i]:
            labels[i] = f"{elements[i]}_slig"
        elif cn[i] < 4:
            labels[i] = f"{elements[i]}_surf"
        else:
            labels[i] = f"{elements[i]}_core"
    return labels, cn


class DFTDataset:
    def __init__(self, dataset_configs, knn_config, first_n_frames=None):
        self.frame_data, self.forces_data = [], []
        print(f"\n{'='*60}\nLOADING DATA\n{'='*60}")
        for ds in dataset_configs:
            name = ds.get('name', ds['xyz'])
            fmt = ds.get('format', 'extxyz')
            n_frames = ds.get('first_n_frames', first_n_frames)
            print(f"\n{name} (format: {fmt})")
            skip = ds.get('skip_frames', None)
            symbols, pos_list, frc_list, cell_bohr = read_trajectory(ds['xyz'], fmt=fmt, first_n=n_frames, skip_n=skip)
            self.frame_data.extend([(symbols, p) for p in pos_list])
            self.forces_data.extend(frc_list)
            if cell_bohr is not None:
                self.cell_bohr = cell_bohr  # store for PBC edge building

        self.knn_config = knn_config
        inorganic = set(knn_config['inorganic_elements'])

        # Assign subtypes from frame 0
        symbols_0 = self.frame_data[0][0]
        pos_0_ang = np.asarray(self.frame_data[0][1]) * BOHR_TO_ANGSTROM
        if knn_config.get('use_subtypes', True):
            self.subtypes, self.cn = assign_subtypes(
                symbols_0, pos_0_ang, inorganic,
                knn_config.get('subtype_ligand_cutoff', 3.0),
                knn_config.get('subtype_inorganic_cutoff', 3.5))
        else:
            # Raw element names, no CN-based subtyping
            self.subtypes = list(symbols_0)
            self.cn = np.zeros(len(symbols_0), dtype=int)

        self.elements = sorted(set(self.subtypes))
        self.element_to_index = {e: i for i, e in enumerate(self.elements)}

        counts = Counter(self.subtypes)
        print(f"\nSubtypes ({len(self.elements)}):")
        for st in self.elements:
            print(f"  {st}: {counts[st]}")

        ne = len(self.elements)
        self.pair_names = [f"{e1}-{e2}" for e1, e2 in combinations_with_replacement(self.elements, 2)]
        self.pair_name_to_index = {pn: i for i, pn in enumerate(self.pair_names)}
        self.pair_lookup = np.zeros((ne, ne), dtype=np.int64)
        for i in range(ne):
            for j in range(ne):
                lo, hi = min(i, j), max(i, j)
                self.pair_lookup[i, j] = self.pair_name_to_index[f"{self.elements[lo]}-{self.elements[hi]}"]
        print(f"Total: {len(self.frame_data)} frames, {len(self.elements)} types, {len(self.pair_names)} pairs")

    def build_graphs(self, shell=1):
        """Build graphs with single cutoff per pair. ALL atoms are triplet centers."""
        pos = np.asarray(self.frame_data[0][1])
        max_neighbors = self.knn_config.get('max_gap_neighbors', 20)
        max_cutoff_ang = self.knn_config.get('max_cutoff', None)
        max_cutoff_bohr = max_cutoff_ang * ANGSTROM_TO_BOHR if max_cutoff_ang else None
        fallback_cutoff = self.knn_config.get('ligand_cutoff', 4.0) * ANGSTROM_TO_BOHR

        subtypes_arr = np.array(self.subtypes)
        type_indices = {t: np.where(subtypes_arr == t)[0] for t in self.elements}

        # ---- Scope SW to the Cd-Se framework + Cd-O/Se-O. The 2-body shape is the fixed
        # II-VI Stillinger-Weber form (Zhou et al. PRB 88, 085309, 2013): the dimensionless
        # ratios A, B, a and the sigma/r0 ratio are taken from that form. The ONLY thing we
        # measure here is the length scale r0 (the RDF first-peak bond length); sigma, the
        # cutoff and A, B then follow from the Zhou ratios (see the per-pair block below).
        INORG = {'Cd', 'Se'}
        base = lambda t: t.partition('_')[0]

        def in_scope(ta, tb):
            ba, bb = base(ta), base(tb)
            cd_se = ba in INORG and bb in INORG and ba != bb
            inorg_o = (ba in INORG and bb == 'O') or (bb in INORG and ba == 'O')
            return cd_se or inorg_o

        sample = range(0, len(self.frame_data), max(1, len(self.frame_data) // 50))

        BOND_MAX = 3.3 * ANGSTROM_TO_BOHR     # a real first-neighbour bond is shorter than this

        def rdf_features(ta, tb, ia, ib):
            """From the pair RDF: first-peak bond length r0, first-minimum r_cut,
            coordination number, and the first-shell bond-length variance."""
            edges = np.linspace(1.8 * ANGSTROM_TO_BOHR, 5.5 * ANGSTROM_TO_BOHR, 75)
            r = 0.5 * (edges[:-1] + edges[1:])
            hist = np.zeros(len(r)); cn = 0.0; bond_d = []
            for fi in sample:
                pos = np.asarray(self.frame_data[fi][1])
                d = cdist(pos[ia], pos[ib])
                if ta == tb:
                    d = d[~np.eye(len(ia), dtype=bool)].reshape(len(ia), -1)
                hist += np.histogram(d.ravel(), bins=edges)[0]
                cn += (d < BOND_MAX).sum() / len(ia)
                dd = d.ravel(); bond_d.append(dd[dd < BOND_MAX])
            g = hist / np.maximum(r ** 2, 1e-12)
            g = np.convolve(g, np.ones(3) / 3, mode='same')
            bond = r < BOND_MAX
            if bond.sum() < 2:
                return None
            r0 = float(r[bond][int(np.argmax(g[bond]))])          # first-peak bond length
            win = (r > r0) & (r < r0 + 2.0 * ANGSTROM_TO_BOHR)
            if win.sum() < 2:
                return None
            r_cut = float(r[win][int(np.argmin(g[win]))])         # first minimum after the peak
            # bond-length fluctuation: variance of the FIRST PEAK only (tight window),
            # not the whole <3.3A range (which would include the broad tail).
            bd = np.concatenate(bond_d)
            peak = bd[np.abs(bd - r0) < 0.35 * ANGSTROM_TO_BOHR]
            var_r = float(peak.var()) if len(peak) > 5 else float('nan')
            return r0, r_cut, cn / len(sample), var_r

        self.cutoffs_bohr, self.sigma_bohr, self.r0_bohr = {}, {}, {}
        self.shape_A, self.shape_B, self.var_r_bohr = {}, {}, {}
        print("\nScoped SW pairs -- INITIAL 2-body shape from the RDF (pass 1; the optional")
        print("curvature pass 2 refines it with the force-matched depth -- self-consistency):")
        for i, ta in enumerate(self.elements):
            for tb in self.elements[i:]:
                if not in_scope(ta, tb):
                    continue
                ia, ib = type_indices[ta], type_indices[tb]
                if len(ia) == 0 or len(ib) == 0:
                    continue
                res = rdf_features(ta, tb, ia, ib)
                if res is None:
                    continue
                r0, r_cut, cn, var_r = res
                if cn < 0.3 or not (r_cut > r0 > 0) or not np.isfinite(var_r) or var_r <= 0:
                    print(f"  {canonical_pair(ta,tb):>12}: not a bond (CN={cn:.2f}) -- skipped")
                    continue
                # 2-body SHAPE = the II-VI Stillinger-Weber form (Zhou et al. PRB 88, 085309,
                # 2013): A,B,a,gamma,p,q are the dimensionless form for this material class
                # (cited like p=4,q=0). The SCALES (sigma from r0, eps from forces, lambda
                # from fluctuations, theta0 from angles) remain data-derived. Deriving the
                # shape itself from the RDF (curvature match) was tried and fails -- the SW
                # form cannot match the ionic well's curvature AND keep a deep enough well
                # (see solve_sw_shape + the fitter's optional curvature_match 2-pass).
                ZHOU_A, ZHOU_B, A_FACTOR, A_CUT = 7.0496, 1.116149, 1.28094, 1.953387
                sigma = r0 / A_FACTOR
                pk = canonical_pair(ta, tb)
                self.r0_bohr[pk] = r0
                self.var_r_bohr[pk] = var_r                   # for the optional curvature 2-pass
                self.sigma_bohr[pk] = sigma
                self.cutoffs_bohr[pk] = A_CUT * sigma         # = a*sigma
                self.shape_A[pk], self.shape_B[pk] = ZHOU_A, ZHOU_B
                print(f"  {pk:>12}: r0={r0*BOHR_TO_ANGSTROM:.3f} sigma={sigma*BOHR_TO_ANGSTROM:.3f} "
                      f"cut={A_CUT*sigma*BOHR_TO_ANGSTROM:.3f} (II-VI SW shape, Zhou 2013)")

        # triplet types: inorganic-centred only (the framework angles)
        scoped_pairs = set(self.cutoffs_bohr.keys())
        all_possible = set()
        for center in self.elements:
            if base(center) not in INORG:
                continue
            nbrs = [t for t in self.elements if canonical_pair(center, t) in scoped_pairs]
            for n1 in nbrs:
                for n2 in nbrs:
                    all_possible.add(canonical_triplet(center, n1, n2))
        self.triplet_type_names = sorted(all_possible)
        self.triplet_type_to_index = {t: i for i, t in enumerate(self.triplet_type_names)}
        print(f"  Scoped pairs: {len(self.cutoffs_bohr)}, triplet types: "
              f"{len(self.triplet_type_names)} -> {self.triplet_type_names}")

        # forces are fit ONLY on inorganic atoms (no CHARMM contribution there)
        self.fit_atom_mask = np.array([base(s) in INORG for s in self.subtypes])

        # Build graphs
        print("\nBuilding graphs...")
        t_start = time.time()
        self.graphs = []
        pair_lookup_t = torch.tensor(self.pair_lookup, dtype=torch.long)
        raw_symbols = self.frame_data[0][0]

        total_triplets = 0
        n_frames = len(self.frame_data)
        for i, (syms, pos_bohr) in enumerate(self.frame_data):
            if i % 500 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 and i > 0 else 0
                eta = (n_frames - i) / rate if rate > 0 else 0
                print(f"  Frame {i}/{n_frames} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

            cell = self.cell_bohr if hasattr(self, 'cell_bohr') else None
            src, tgt = build_edges(pos_bohr, syms, self.subtypes, self.cutoffs_bohr, cell_bohr=cell)
            if len(src) == 0:
                continue

            pos_t = torch.tensor(pos_bohr, dtype=torch.float64)
            src_t = torch.tensor(src, dtype=torch.long)
            tgt_t = torch.tensor(tgt, dtype=torch.long)
            edge_index = torch.stack([src_t, tgt_t])
            disp = pos_t[tgt_t] - pos_t[src_t]
            dist = torch.norm(disp, dim=1)
            unit_vec = disp / dist.clamp(min=1e-10).unsqueeze(1)
            elem_idx = torch.tensor([self.element_to_index[s] for s in self.subtypes], dtype=torch.long)
            pair_indices = pair_lookup_t[elem_idx[src_t], elem_idx[tgt_t]]

            frc = self.forces_data[i]
            dft_forces = torch.tensor(frc, dtype=torch.float64) if frc is not None else torch.zeros_like(pos_t)

            tri_a, tri_c, tri_b, pi_ca, pi_cb, tri_type_idx = build_triplets(
                edge_index, elem_idx, pair_lookup_t,
                self.triplet_type_to_index, self.subtypes)
            total_triplets += len(tri_a)

            # Precompute triplet geometry
            if len(tri_a) > 0:
                vec_ca = pos_t[tri_a] - pos_t[tri_c]
                vec_cb = pos_t[tri_b] - pos_t[tri_c]
                r_ca = torch.norm(vec_ca, dim=1).clamp(min=1e-10)
                r_cb = torch.norm(vec_cb, dim=1).clamp(min=1e-10)
                cos_theta = ((vec_ca * vec_cb).sum(1) / (r_ca * r_cb)).clamp(-1, 1)
                rc_lookup = torch.full((len(self.pair_names),), 1e10, dtype=torch.float64)
                for pn, cut in self.cutoffs_bohr.items():
                    if pn in self.pair_name_to_index:
                        rc_lookup[self.pair_name_to_index[pn]] = cut
                rc_ca = rc_lookup[pi_ca]; rc_cb = rc_lookup[pi_cb]
                in_ca = (r_ca < rc_ca).to(torch.float64)
                in_cb = (r_cb < rc_cb).to(torch.float64)
                d_ca = (r_ca - rc_ca).clamp(max=-1e-8)
                d_cb = (r_cb - rc_cb).clamp(max=-1e-8)
            else:
                e3 = torch.empty(0, 3, dtype=torch.float64)
                e1 = torch.empty(0, dtype=torch.float64)
                vec_ca = vec_cb = e3
                cos_theta = in_ca = in_cb = e1
                d_ca = d_cb = e1

            self.graphs.append(SWData(
                pos=pos_t, edge_index=edge_index,
                fit_mask=torch.tensor(self.fit_atom_mask, dtype=torch.bool),
                distances=dist, edge_unit_vectors=unit_vec,
                element_indices=elem_idx, pair_indices=pair_indices,
                dft_forces=dft_forces,
                triplet_a=tri_a, triplet_center=tri_c, triplet_b=tri_b,
                triplet_pi_ca=pi_ca, triplet_pi_cb=pi_cb,
                triplet_type_idx=tri_type_idx,
                tri_vec_ca=vec_ca, tri_vec_cb=vec_cb, tri_cos_theta=cos_theta,
                tri_in_ca=in_ca, tri_in_cb=in_cb,
                tri_rainv_ca=1.0 / d_ca if len(d_ca) > 0 else e1,
                tri_rainv_cb=1.0 / d_cb if len(d_cb) > 0 else e1,
                tri_cos_over_rsq_ca=cos_theta / r_ca**2 if len(r_ca) > 0 else e1,
                tri_cos_over_rsq_cb=cos_theta / r_cb**2 if len(r_cb) > 0 else e1,
                tri_cross_rinv=1.0 / (r_ca * r_cb) if len(r_ca) > 0 else e1,
                tri_rainvsq_over_r_ca=(1.0 / d_ca)**2 / r_ca if len(d_ca) > 0 else e1,
                tri_rainvsq_over_r_cb=(1.0 / d_cb)**2 / r_cb if len(d_cb) > 0 else e1,
            ))

        avg_tri = total_triplets / max(len(self.graphs), 1)
        print(f"done ({len(self.graphs)} graphs, avg {avg_tri:.0f} triplets/frame, "
              f"{time.time() - t_start:.0f}s)")
