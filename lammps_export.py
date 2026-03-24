import os
import numpy as np
from datetime import datetime
from sklearn.mixture import GaussianMixture

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, canonical_pair, base_element, MASSES,
    has_bonds_in_data,
)


def classify_atoms_embedding(embeddings_multi, atom_symbols, inorganic_elements,
                             n_clusters_per_element=None, max_clusters=8):
    symbols = np.array(atom_symbols)
    inorganic_set = set(inorganic_elements)

    if embeddings_multi.ndim == 3:
        avg_emb = np.mean(embeddings_multi, axis=0)
    else:
        avg_emb = embeddings_multi

    atom_types = list(atom_symbols)
    type_legend = {}

    for elem in sorted(inorganic_set):
        mask = symbols == elem
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        elem_emb = avg_emb[indices]

        if n_clusters_per_element and elem in n_clusters_per_element:
            n_c = n_clusters_per_element[elem]
        else:
            n_c = _select_n_clusters_bic(elem_emb, max_clusters)

        if n_c <= 1 or len(indices) <= n_c:
            type_name = f"{elem}_0"
            type_legend[type_name] = {
                'count': len(indices),
                'description': f"{elem}, single cluster ({len(indices)} atoms)",
            }
            for idx in indices:
                atom_types[idx] = type_name
            continue

        gmm = GaussianMixture(n_components=n_c, covariance_type='full',
                               n_init=5, random_state=42)
        labels = gmm.fit_predict(elem_emb)

        unique_labels, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)

        label_to_name = {}
        for rank, orig_label in enumerate(order):
            type_name = f"{elem}_{rank}"
            count = counts[orig_label]
            label_to_name[orig_label] = type_name
            type_legend[type_name] = {
                'count': int(count),
                'description': f"{elem}, embedding cluster {rank} ({count} atoms)",
            }

        for local_i, global_i in enumerate(indices):
            atom_types[global_i] = label_to_name[labels[local_i]]

    return atom_types, type_legend


def _select_n_clusters_bic(X, max_k):
    n = len(X)
    if n <= 2:
        return 1
    best_k, best_bic = 1, np.inf
    for k in range(1, min(max_k + 1, n)):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                   n_init=3, random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            break
    return best_k


def compute_discretization_error(edge_index, atom_types, per_edge_params,
                                 pair_indices, pair_names, frozen_pairs, graph):
    frozen_set = set(frozen_pairs)
    edge_src = edge_index[0]
    n_atoms = len(atom_types)

    keep_mask = np.array([pair_names[pi] not in frozen_set for pi in pair_indices])
    src_types = np.array([atom_types[i] for i in edge_src])
    tgt_types = np.array([atom_types[i] for i in edge_index[1]])
    cp_names = np.array([canonical_pair(s, t) for s, t in zip(src_types, tgt_types)])

    cp_medians = {}
    for cp in np.unique(cp_names[keep_mask]):
        mask = (cp_names == cp) & keep_mask
        cp_medians[cp] = {
            'D_e': float(np.median(per_edge_params['D_e'][mask])),
            'alpha': float(np.median(per_edge_params['alpha'][mask])),
            'r0': float(np.median(per_edge_params['r0'][mask])),
        }

    D_e_disc = np.copy(per_edge_params['D_e'])
    alpha_disc = np.copy(per_edge_params['alpha'])
    r0_disc = np.copy(per_edge_params['r0'])

    for i in range(len(edge_src)):
        cp = cp_names[i]
        if cp in cp_medians:
            D_e_disc[i] = cp_medians[cp]['D_e']
            alpha_disc[i] = cp_medians[cp]['alpha']
            r0_disc[i] = cp_medians[cp]['r0']
        elif not keep_mask[i]:
            pass
        else:
            D_e_disc[i] = 0.001

    distances_au = graph.distances.numpy()
    unit_vecs = graph.edge_unit_vectors.numpy()
    symbols = [atom_types[i].partition('_')[0] if '_' in atom_types[i] else atom_types[i]
               for i in range(n_atoms)]
    inorganic_mask = np.array([s not in ('C', 'H', 'O') for s in symbols])

    def _compute_forces(D_e_ev, alpha_invang, r0_ang):
        D_e_au = D_e_ev / HARTREE_TO_EV
        alpha_au = alpha_invang * BOHR_TO_ANGSTROM
        r0_au = r0_ang / BOHR_TO_ANGSTROM
        x = distances_au - r0_au
        exp1 = np.exp(-alpha_au * x)
        sf = 2.0 * D_e_au * alpha_au * (exp1**2 - exp1)
        fvecs = -sf[:, None] * unit_vecs
        forces = np.zeros((n_atoms, 3))
        np.add.at(forces, edge_src, fvecs)
        return forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

    cont_forces = _compute_forces(per_edge_params['D_e'],
                                  per_edge_params['alpha'],
                                  per_edge_params['r0'])
    disc_forces = _compute_forces(D_e_disc, alpha_disc, r0_disc)
    dft_evang = graph.dft_forces.numpy() * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

    cont_rmse = np.sqrt(np.mean((cont_forces[inorganic_mask] - dft_evang[inorganic_mask])**2))
    disc_rmse = np.sqrt(np.mean((disc_forces[inorganic_mask] - dft_evang[inorganic_mask])**2))

    return cont_rmse, disc_rmse


def _collect_embeddings(model_cpu, dataset, n_frames_emb=50):
    n_use = min(n_frames_emb, len(dataset.graphs))
    all_emb = []
    for fi in range(n_use):
        emb = model_cpu.get_embeddings(dataset.graphs[fi])
        if emb is not None:
            all_emb.append(emb)
    return np.stack(all_emb), n_use


def _distribute_k(total_k, elem_counts):
    total_inorg = sum(elem_counts.values())
    n_per_elem = {}
    for elem in sorted(elem_counts.keys()):
        frac = elem_counts[elem] / total_inorg
        n_per_elem[elem] = max(1, round(total_k * frac))
    while sum(n_per_elem.values()) > total_k:
        biggest = max(n_per_elem, key=n_per_elem.get)
        if n_per_elem[biggest] > 1:
            n_per_elem[biggest] -= 1
    while sum(n_per_elem.values()) < total_k:
        smallest = min(n_per_elem, key=n_per_elem.get)
        n_per_elem[smallest] += 1
    return n_per_elem


def _evaluate_clustering(atom_types, edge_index_np, per_edge_params,
                         pair_indices, pair_names, frozen_pairs, graph, symbols):
    cp_data = _build_cluster_pair_edges(
        edge_index_np, atom_types, per_edge_params,
        pair_indices, pair_names, frozen_pairs,
        edge_distances_angstrom=graph.distances.numpy() * BOHR_TO_ANGSTROM)
    n_fallback = sum(1 for d in cp_data.values() if d.get('is_fallback'))
    n_pairs = len(cp_data)
    _, disc_rmse = compute_discretization_error(
        edge_index_np, atom_types, per_edge_params,
        pair_indices, pair_names, frozen_pairs, graph)
    n_inorg_types = sum(1 for t in set(atom_types) if t not in set(symbols))
    return disc_rmse, n_pairs, n_fallback, n_inorg_types


def auto_select_k(avg_emb, symbols, inorganic, edge_index_np, per_edge_params,
                  pair_indices, pair_names, frozen_pairs, graph,
                  max_k=12, verbose=True):
    sym_arr = np.array(symbols)
    inorganic_set = set(inorganic)
    elem_counts = {e: int(np.sum(sym_arr == e)) for e in inorganic}

    if verbose:
        print(f"\n  Auto-selecting cluster count (k=2..{max_k})...")

    best_k = 2
    best_rmse = np.inf
    best_types = None
    best_legend = None

    for total_k in range(2, max_k + 1):
        n_per_elem = _distribute_k(total_k, elem_counts)
        atom_types, type_legend = classify_atoms_embedding(
            avg_emb, symbols, inorganic_set,
            n_clusters_per_element=n_per_elem)
        disc_rmse, n_pairs, n_fallback, n_inorg = _evaluate_clustering(
            atom_types, edge_index_np, per_edge_params,
            pair_indices, pair_names, frozen_pairs, graph, symbols)

        if verbose:
            print(f"  k={total_k:>2}: RMSE={disc_rmse:.4f}, {n_pairs} pairs ({n_fallback} fallback)")

        if disc_rmse < best_rmse:
            best_rmse = disc_rmse
            best_k = total_k
            best_types = atom_types
            best_legend = type_legend

    if verbose:
        print(f"  -> Best: k={best_k} (RMSE={best_rmse:.4f})")

    return best_k, best_types, best_legend


def classify_atoms(edge_index, atom_symbols, inorganic_elements):
    symbols = np.array(atom_symbols)
    inorganic_set = set(inorganic_elements)
    inorganic_sorted = sorted(inorganic_set)

    is_inorganic = np.array([s in inorganic_set for s in symbols])
    src, tgt = edge_index[0], edge_index[1]
    n_atoms = len(symbols)

    neighbor_counts = [{} for _ in range(n_atoms)]
    has_organic_neighbor = np.zeros(n_atoms, dtype=bool)

    for s, t in zip(src, tgt):
        tgt_sym = symbols[t]
        if tgt_sym in inorganic_set:
            neighbor_counts[s][tgt_sym] = neighbor_counts[s].get(tgt_sym, 0) + 1
        elif is_inorganic[s]:
            has_organic_neighbor[s] = True

    sig_to_atoms = {}
    for i in range(n_atoms):
        if not is_inorganic[i]:
            continue
        elem = symbols[i]
        parts = []
        for nb_elem in inorganic_sorted:
            if nb_elem == elem:
                continue
            count = neighbor_counts[i].get(nb_elem, 0)
            if count > 0:
                parts.append(f"{count}{nb_elem}")
        inorg_sig = '_'.join(parts) if parts else 'isolated'
        key = (elem, inorg_sig, bool(has_organic_neighbor[i]))
        sig_to_atoms.setdefault(key, []).append(i)

    base_groups = {}
    for (elem, inorg_sig, has_lig), atoms in sig_to_atoms.items():
        base_groups.setdefault((elem, inorg_sig), {})[has_lig] = atoms

    atom_types = list(atom_symbols)
    type_legend = {}

    for (elem, inorg_sig), lig_groups in sorted(base_groups.items()):
        needs_lig_suffix = len(lig_groups) > 1

        for has_lig, atom_indices in sorted(lig_groups.items()):
            type_name = f"{elem}_{inorg_sig}"
            if needs_lig_suffix and has_lig:
                type_name += "_lig"

            desc = f"{elem}, {inorg_sig} inorganic coord"
            if needs_lig_suffix:
                desc += ", has ligand nbrs" if has_lig else ", no ligand nbrs"
            elif has_lig:
                desc += ", has ligand nbrs"

            type_legend[type_name] = {
                'count': len(atom_indices),
                'description': desc,
            }
            for idx in atom_indices:
                atom_types[idx] = type_name

    return atom_types, type_legend


def _build_cluster_pair_edges(edge_index, atom_types, per_edge_params,
                              pair_indices, pair_names, frozen_pairs,
                              edge_distances_angstrom=None):
    edge_src = edge_index[0]
    edge_tgt = edge_index[1]

    frozen_set = set(frozen_pairs)
    keep_mask = np.array([pair_names[pi] not in frozen_set for pi in pair_indices])

    src_types = np.array([atom_types[i] for i in edge_src[keep_mask]])
    tgt_types = np.array([atom_types[i] for i in edge_tgt[keep_mask]])
    cp_names = np.array([canonical_pair(s, t) for s, t in zip(src_types, tgt_types)])
    base_pairs = np.array([pair_names[pi] for pi in pair_indices[keep_mask]])
    De_vals = per_edge_params['D_e'][keep_mask]
    alpha_vals = per_edge_params['alpha'][keep_mask]
    r0_vals = per_edge_params['r0'][keep_mask]
    dist_vals = edge_distances_angstrom[keep_mask] if edge_distances_angstrom is not None else None

    result = {}
    base_pair_params = {}

    for cp in np.unique(cp_names):
        mask = cp_names == cp
        D_arr = De_vals[mask]
        a_arr = alpha_vals[mask]
        r_arr = r0_vals[mask]
        bp = base_pairs[mask][0]

        result[cp] = {
            'D_e': float(np.median(D_arr)),
            'alpha': float(np.median(a_arr)),
            'r0': float(np.median(r_arr)),
            'D_e_std': float(np.std(D_arr)),
            'alpha_std': float(np.std(a_arr)),
            'r0_std': float(np.std(r_arr)),
            'n_edges': len(D_arr),
            'base_pair': bp,
            'is_fallback': False,
            'max_edge_dist': float(np.max(dist_vals[mask])) if dist_vals is not None else 0.0,
        }

        if bp not in base_pair_params:
            base_pair_params[bp] = {'D_e': [], 'alpha': [], 'r0': []}
        base_pair_params[bp]['D_e'].append(D_arr)
        base_pair_params[bp]['alpha'].append(a_arr)
        base_pair_params[bp]['r0'].append(r_arr)

    base_pair_medians = {}
    for bp, arrays in base_pair_params.items():
        base_pair_medians[bp] = {
            'D_e': float(np.median(np.concatenate(arrays['D_e']))),
            'alpha': float(np.median(np.concatenate(arrays['alpha']))),
            'r0': float(np.median(np.concatenate(arrays['r0']))),
        }

    # Fill missing cluster-pair combinations with near-zero D_e
    cluster_types = sorted(set(atom_types))
    for i, ct1 in enumerate(cluster_types):
        for ct2 in cluster_types[i:]:
            cp_name = canonical_pair(ct1, ct2)
            if cp_name in result:
                continue
            bp = canonical_pair(base_element(ct1), base_element(ct2))
            if bp in frozen_set or bp not in base_pair_medians:
                continue
            med = base_pair_medians[bp]
            result[cp_name] = {
                'D_e': 0.001, 'alpha': med['alpha'], 'r0': med['r0'],
                'D_e_std': 0.0, 'alpha_std': 0.0, 'r0_std': 0.0,
                'n_edges': 0, 'base_pair': bp,
                'is_fallback': True, 'max_edge_dist': 0.0,
            }

    return result


def write_lammps_table(cluster_pair_data, filepath,
                       default_cutoff=6.0, n_points=5000, r_min=0.5):
    trained_sections = []

    with open(filepath, 'w') as f:
        f.write(f"# LAMMPS pair_style table: GNN-Morse environment-aware potential\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Each section: cluster-pair-averaged Morse V(r) and F(r)\n\n")

        for cp_name in sorted(cluster_pair_data.keys()):
            data = cluster_pair_data[cp_name]
            D_e, alpha, r0 = data['D_e'], data['alpha'], data['r0']
            section_name = cp_name.replace('-', '_')
            r_arr = np.linspace(r_min, default_cutoff, n_points)
            trained_sections.append(cp_name)

            max_dist = data.get('max_edge_dist', 0.0)
            train_cut = data.get('training_cutoff', 0.0)
            f.write(f"# {cp_name} (from {data['base_pair']}): "
                    f"D_e={D_e:.6f}+/-{data['D_e_std']:.4f} eV, "
                    f"alpha={alpha:.4f}+/-{data['alpha_std']:.4f} /A, "
                    f"r0={r0:.4f}+/-{data['r0_std']:.4f} A "
                    f"({data['n_edges']} edges, max_dist={max_dist:.2f}, "
                    f"training_cutoff={train_cut:.2f})\n")
            f.write(f"{section_name}\n")
            f.write(f"N {n_points} R {r_min} {default_cutoff}\n\n")

            x = r_arr - r0
            exp1 = np.exp(-alpha * x)
            V = D_e * (1.0 - exp1) ** 2
            F = 2.0 * D_e * alpha * (exp1**2 - exp1)
            f.writelines(
                f"{j} {r:.10f} {v:.10e} {fv:.10e}\n"
                for j, r, v, fv in zip(range(1, n_points + 1), r_arr, V, F))
            f.write("\n")

    print(f"Saved: {filepath} ({len(trained_sections)} cluster-pair tables)")
    return trained_sections


def write_lammps_snippet(cluster_elements, trained_sections, frozen_pairs_config,
                         table_filename, filepath, cutoff=10.0, n_points=5000):
    elem_to_type = {e: i + 1 for i, e in enumerate(cluster_elements)}

    base_to_clusters = {}
    for ce in cluster_elements:
        base_to_clusters.setdefault(base_element(ce), []).append(ce)

    with open(filepath, 'w') as f:
        f.write(f"# LAMMPS input snippet for GNN-Morse potential\n")
        f.write(f"# {len(cluster_elements)} atom types: {', '.join(cluster_elements)}\n")
        f.write(f"# Type mapping: {', '.join(f'{e}={i+1}' for i, e in enumerate(cluster_elements))}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        has_frozen = bool(frozen_pairs_config)
        if has_frozen:
            f.write(f"pair_style  hybrid/overlay table linear {n_points} "
                    f"lj/cut {cutoff:.1f}\n\n")
        else:
            f.write(f"pair_style  table linear {n_points}\n\n")

        f.write("# ── GNN-TRAINED MORSE PAIRS (tabulated) ──\n")
        for cp_name in sorted(trained_sections):
            e1, e2 = cp_name.split('-')
            t1, t2 = elem_to_type.get(e1), elem_to_type.get(e2)
            if t1 is None or t2 is None:
                continue
            section_name = cp_name.replace('-', '_')
            style = "table " if has_frozen else ""
            f.write(f"pair_coeff  {t1} {t2} {style}"
                    f"{table_filename} {section_name}  # {cp_name}\n")

        if has_frozen:
            f.write(f"\n# ── FROZEN ORGANIC LJ PAIRS ──\n")
            for pn, vals in sorted(frozen_pairs_config.items()):
                base_e1, base_e2 = pn.split('-')
                variants_1 = base_to_clusters.get(base_e1, [])
                variants_2 = base_to_clusters.get(base_e2, [])
                if not variants_1 or not variants_2:
                    continue
                eps = vals['D_e']
                sig = vals['r0'] / (2**(1/6))
                for v1 in variants_1:
                    for v2 in variants_2:
                        t1, t2 = elem_to_type[v1], elem_to_type[v2]
                        if t1 > t2:
                            t1, t2 = t2, t1
                        f.write(f"pair_coeff  {t1} {t2} lj/cut "
                                f"{eps:.8f} {sig:.5f}  # {canonical_pair(v1, v2)} ({pn})\n")

    print(f"Saved: {filepath}")


def _write_lammps_in(filepath, cluster_elements, has_bonds=True, temp=300, nsteps=10000):
    elem_names = [base_element(ce) for ce in cluster_elements]
    with open(filepath, 'w') as f:
        f.write("# GNN-Morse MD\n")
        f.write("units           metal\n")
        if has_bonds:
            f.write("atom_style      full\n")
            f.write("improper_style  harmonic\n")
            f.write("bond_style      harmonic\n")
            f.write("angle_style     harmonic\n")
        else:
            f.write("atom_style      charge\n")
        f.write("boundary        f f f\n\n")
        if has_bonds:
            f.write("special_bonds   charmm\n")
        f.write("read_data       gnn_morse.data\n")
        if has_bonds:
            f.write("special_bonds   charmm\n")
        f.write("\ninclude         potentials.txt\n")
        f.write(f"""
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
timestep        0.001

thermo          100
thermo_style    custom step temp pe ke etotal press
log             gnn_morse.log

dump            1 all custom 1 gnn_morse.xyz element xu yu zu vx vy vz fx fy fz
dump_modify     1 element {' '.join(elem_names)}
dump_modify     1 sort id

velocity        all create {temp} 12345
fix             1 all nve
fix             2 all temp/csvr {temp}.0 {temp}.0 0.1 54321

run             {nsteps}
""")
    print(f"Wrote {filepath}")


def _to_gen_params(cluster_pair_data):
    gen_params = {}
    for cp_name, d in cluster_pair_data.items():
        gen_params[cp_name] = {
            'D_e': d['D_e'], 'alpha': d['alpha'], 'r0': d['r0'],
            'n': d['n_edges'], 'base': d['base_pair'],
            'fallback': d.get('is_fallback', False),
            'max_edge_dist': d.get('max_edge_dist', 0),
            'training_cutoff': d.get('training_cutoff', 0),
        }
    return gen_params


def export_lammps(model, dataset, config, output_dir,
                  clustering='embedding', n_clusters=None, n_frames_emb=50,
                  original_data=None, temp=300, write_lammps_in=True):
    import torch

    print(f"\n{'='*60}")
    print("LAMMPS EXPORT")
    print('='*60)

    graph = dataset.graphs[0]
    model_cpu = model.cpu()
    model_cpu.eval()

    per_edge_params = model_cpu.get_per_edge_params(graph)
    symbols = dataset.frame_data[0][0]
    inorganic = sorted(set(config.get('knn_edges', {}).get('inorganic_elements', ['Cd', 'Se'])))
    inorganic_set = set(inorganic)
    edge_index_np = graph.edge_index.numpy()

    # ── Atom classification ──
    if clustering == 'embedding':
        embeddings_multi, n_use = _collect_embeddings(model_cpu, dataset, n_frames_emb)
        avg_emb = np.mean(embeddings_multi, axis=0)
        print(f"\nCollected embeddings from {n_use} frames (shape: {avg_emb.shape})")

        if n_clusters is None:
            frozen_pairs = set(config.get('fixed_pairs', {}).keys())
            pair_indices = graph.pair_indices.numpy()
            best_k, atom_types, type_legend = auto_select_k(
                avg_emb, symbols, inorganic, edge_index_np, per_edge_params,
                pair_indices, dataset.pair_names, frozen_pairs, graph,
                max_k=12, verbose=True)
            print(f"\nAtom classification (embedding-based GMM, auto k={best_k}):")
        else:
            n_clusters_per_elem = None
            if isinstance(n_clusters, dict):
                n_clusters_per_elem = n_clusters
            elif isinstance(n_clusters, int):
                sym_arr = np.array(symbols)
                elem_counts = {e: int(np.sum(sym_arr == e)) for e in inorganic}
                n_clusters_per_elem = _distribute_k(n_clusters, elem_counts)
            atom_types, type_legend = classify_atoms_embedding(
                embeddings_multi, symbols, inorganic_set,
                n_clusters_per_element=n_clusters_per_elem)
            print(f"\nAtom classification (embedding-based GMM, k={n_clusters}):")
    else:
        atom_types, type_legend = classify_atoms(
            edge_index_np, symbols, inorganic_set)
        print(f"\nAtom classification (coordination-based):")

    # Print type legend
    max_name = max(len(t) for t in type_legend) if type_legend else 10
    print(f"  {'Type':<{max_name}}  Count  Description")
    print(f"  {'-'*max_name}  -----  -----------")
    for tname in sorted(type_legend.keys()):
        info = type_legend[tname]
        print(f"  {tname:<{max_name}}  {info['count']:>5}  {info['description']}")

    cluster_elements = sorted(set(atom_types))

    # Print per-edge parameter statistics
    print("\nPer-edge Morse parameters by original pair type (median +/- std):")
    for pi, pn in enumerate(dataset.pair_names):
        mask = graph.pair_indices.numpy() == pi
        if mask.sum() == 0:
            continue
        D = per_edge_params['D_e'][mask]
        a = per_edge_params['alpha'][mask]
        r = per_edge_params['r0'][mask]
        print(f"  {pn:>8}: D_e={np.median(D):.4f}+/-{np.std(D):.4f} eV, "
              f"alpha={np.median(a):.4f}+/-{np.std(a):.4f} /A, "
              f"r0={np.median(r):.4f}+/-{np.std(r):.4f} A  "
              f"({mask.sum()} edges)")

    # ── Build cluster-pair-averaged parameters ──
    frozen_pairs = set(config.get('fixed_pairs', {}).keys())
    fixed_pairs_config = config.get('fixed_pairs', {})

    pair_indices = graph.pair_indices.numpy()
    edge_dists_ang = graph.distances.numpy() * BOHR_TO_ANGSTROM
    cluster_pair_data = _build_cluster_pair_edges(
        edge_index_np, atom_types, per_edge_params,
        pair_indices, dataset.pair_names, frozen_pairs,
        edge_distances_angstrom=edge_dists_ang)

    # Attach training cutoffs
    gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
    if gap_cuts:
        print(f"\nTraining cutoffs (gap_cut per base pair):")
        for bp in sorted(gap_cuts):
            print(f"  {bp}: {gap_cuts[bp]:.2f} A")
    for cp_name, d in cluster_pair_data.items():
        d['training_cutoff'] = gap_cuts.get(d['base_pair'], 0.0)

    # Print cluster pair statistics
    n_fallback = sum(1 for d in cluster_pair_data.values() if d.get('is_fallback'))
    n_trained = len(cluster_pair_data) - n_fallback
    print(f"\nCluster-pair Morse parameters ({len(cluster_pair_data)} pairs: "
          f"{n_trained} trained, {n_fallback} fallback):")
    for cp_name in sorted(cluster_pair_data.keys()):
        d = cluster_pair_data[cp_name]
        tag = " [FALLBACK]" if d.get('is_fallback') else ""
        max_d = d.get('max_edge_dist', 0)
        tc = d.get('training_cutoff', 0)
        print(f"  {cp_name:>22}: D_e={d['D_e']:.4f}+/-{d['D_e_std']:.4f} eV, "
              f"alpha={d['alpha']:.4f}+/-{d['alpha_std']:.4f} /A, "
              f"r0={d['r0']:.4f}+/-{d['r0_std']:.4f} A  "
              f"({d['n_edges']} edges, max={max_d:.2f}A, cutoff={tc:.2f}A, "
              f"from {d['base_pair']}){tag}")

    # Diagnose fallback pairs
    if n_fallback > 0:
        print(f"\n  ── Fallback Diagnosis ({n_fallback} pairs) ──")
        organic_bases = {'C', 'H', 'O'}
        expected, surprising = [], []
        for cp_name, d in sorted(cluster_pair_data.items()):
            if not d.get('is_fallback'):
                continue
            ct1, ct2 = cp_name.split('-')
            b1, b2 = base_element(ct1), base_element(ct2)
            inorg1, inorg2 = b1 not in organic_bases, b2 not in organic_bases
            if (inorg1 != inorg2) or (inorg1 and inorg2 and b1 == b2):
                expected.append(f"    {cp_name:>30} ({d['base_pair']})")
            else:
                surprising.append(f"    {cp_name:>30} ({d['base_pair']})")
        if expected:
            print(f"  Expected (0 edges OK): {len(expected)}")
            for line in expected:
                print(line)
        if surprising:
            print(f"  Surprising (may need investigation): {len(surprising)}")
            for line in surprising:
                print(line)

    # ── Write output files ──
    os.makedirs(output_dir, exist_ok=True)

    # Atom type mapping (always needed for .data remapping)
    mapping_path = os.path.join(output_dir, 'atom_type_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write(f"# Atom type mapping\n")
        f.write(f"# atom_index(1-based)  lammps_type  atom_type  original_element\n")
        f.write(f"# LAMMPS types: {', '.join(f'{e}={i+1}' for i, e in enumerate(cluster_elements))}\n")
        for i, (ct, orig) in enumerate(zip(atom_types, symbols)):
            lammps_type = cluster_elements.index(ct) + 1
            f.write(f"{i+1}  {lammps_type}  {ct}  {orig}\n")
    print(f"Saved: {mapping_path}")

    if write_lammps_in:
        # Full export: table + snippet + .data + .in + potentials/
        table_path = os.path.join(output_dir, 'gnn_morse.table')
        trained_sections = write_lammps_table(cluster_pair_data, table_path)
        write_lammps_snippet(cluster_elements, trained_sections, fixed_pairs_config,
                             'gnn_morse.table',
                             os.path.join(output_dir, 'gnn_morse_snippet.txt'))

        if original_data and os.path.exists(original_data):
            from .generate_lammps_files import (
                remap_data_file, write_lammps_input, write_potential_file,
                add_missing_pairs,
            )
            gen_params = add_missing_pairs(_to_gen_params(cluster_pair_data), cluster_elements)
            has_bonds = has_bonds_in_data(original_data)
            print(f"\nDirect LAMMPS export: original_data={original_data}, "
                  f"has_bonds={has_bonds}, temp={temp}")

            remap_data_file(original_data, mapping_path,
                            os.path.join(output_dir, 'gnn_morse.data'),
                            cluster_elements)
            write_lammps_input(gen_params, os.path.join(output_dir, 'gnn_morse.in'),
                               cluster_elements, temp=temp, has_bonds=has_bonds)
            pot_dir = os.path.join(output_dir, 'potentials')
            os.makedirs(pot_dir, exist_ok=True)
            write_potential_file(gen_params, os.path.join(pot_dir, 'gnn_morse.txt'))
            import shutil
            shutil.copy2(table_path, pot_dir)
    else:
        # Minimal export: mapping + .data + potentials.txt + gnn_morse.in
        if original_data and os.path.exists(original_data):
            from .generate_lammps_files import (
                remap_data_file, write_potentials_file, add_missing_pairs,
            )
            gen_params = add_missing_pairs(_to_gen_params(cluster_pair_data), cluster_elements)
            has_bonds = has_bonds_in_data(original_data)

            remap_data_file(original_data, mapping_path,
                            os.path.join(output_dir, 'gnn_morse.data'),
                            cluster_elements)
            write_potentials_file(gen_params, os.path.join(output_dir, 'potentials.txt'),
                                  cluster_elements, has_bonds=has_bonds)
            _write_lammps_in(os.path.join(output_dir, 'gnn_morse.in'),
                             cluster_elements, has_bonds=has_bonds, temp=temp)

    return atom_types, type_legend, cluster_pair_data, cluster_elements
