import os
import numpy as np

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, canonical_pair, base_element,
    has_bonds_in_data,
)


def _describe_vq_types(atom_types, symbols, edge_index_np, inorganic_elements):
    # Label VQ codes as core/surf by coordination ranking.
    # Highest cross-species coordination = core, rest = surf.
    inorganic_set = set(inorganic_elements)
    n_atoms = len(symbols)
    src, tgt = edge_index_np[0], edge_index_np[1]

    # Per-atom neighbor counts
    cross_coord = np.zeros(n_atoms, dtype=int)
    total_inorg_coord = np.zeros(n_atoms, dtype=int)
    has_ligand = np.zeros(n_atoms, dtype=bool)

    for s, t in zip(src, tgt):
        if symbols[s] in inorganic_set:
            if symbols[t] in inorganic_set:
                total_inorg_coord[s] += 1
                if symbols[t] != symbols[s]:
                    cross_coord[s] += 1
            else:
                has_ligand[s] = True

    # Group VQ codes by element
    unique_types = sorted(set(atom_types))
    elem_codes = {}  # {elem: [code_name, ...]}
    for tname in unique_types:
        elem = tname.partition('_')[0]
        elem_codes.setdefault(elem, []).append(tname)

    label_map = {}

    for elem, codes in elem_codes.items():
        # Non-inorganic: bare element name
        if elem not in inorganic_set:
            for tname in codes:
                label_map[tname] = elem if len(codes) == 1 else tname
            continue

        # Single code: just element name
        if len(codes) == 1:
            label_map[codes[0]] = elem
            continue

        # Multiple codes: rank by coordination to assign core/surf
        scores = []
        for tname in codes:
            idx = [i for i, t in enumerate(atom_types) if t == tname]
            mean_cross = float(np.mean(cross_coord[idx]))
            mean_total = float(np.mean(total_inorg_coord[idx]))
            frac_lig = float(np.mean(has_ligand[idx]))
            # Composite: higher cross, higher total, less ligand → core
            scores.append((mean_cross, mean_total, -frac_lig, tname))

        # Sort descending — highest score = core
        scores.sort(reverse=True)
        label_map[scores[0][3]] = f"{elem}_core"
        for _, _, _, tname in scores[1:]:
            label_map[tname] = f"{elem}_surf"

    return label_map


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
            'n': len(D_arr),
            'base': bp,
            'fallback': False,
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
                'n': 0, 'base': bp,
                'fallback': True, 'max_edge_dist': 0.0,
            }

    return result


def export_lammps(model, dataset, config, output_dir,
                  original_data=None, temp=300):
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
    edge_index_np = graph.edge_index.numpy()

    # Atom classification
    vq_result = model_cpu.get_atom_types(graph)

    if vq_result is not None:
        # VQ path: atom types from trained codebook, relabeled by coordination
        vq_indices, atom_types = vq_result

        # Rename VQ codes to descriptive labels (e.g. Cd_0 -> Cd_core_4Se)
        label_map = _describe_vq_types(atom_types, symbols, edge_index_np, inorganic)
        atom_types = [label_map[t] for t in atom_types]

        type_legend = {}
        for old_name, new_name in sorted(set(label_map.items())):
            count = sum(1 for t in atom_types if t == new_name)
            elem = old_name.partition('_')[0]
            type_legend[new_name] = {
                'count': count,
                'description': f"{elem}, {new_name} ({count} atoms)",
            }

        active_codes = model_cpu.vq.get_active_codes()
        active_str = ', '.join(f"{e}={n}" for e, n in active_codes.items()
                               if model_cpu.vq.n_codes_per_elem[e] > 1)
        print(f"\nAtom classification (VQ codebook, active codes: {active_str}):")

    else:
        # No VQ: each element is one atom type (no subtypes)
        atom_types = list(symbols)
        type_legend = {}
        for elem in sorted(set(symbols)):
            count = sum(1 for s in symbols if s == elem)
            type_legend[elem] = {'count': count, 'description': f"{elem} ({count} atoms)"}
        print(f"\nAtom classification (element-based, no subtypes):")

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

    # Build cluster-pair-averaged parameters
    frozen_pairs = set(config.get('fixed_pairs', {}).keys())

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
        d['training_cutoff'] = gap_cuts.get(d['base'], 0.0)

    # Print cluster pair statistics
    n_fallback = sum(1 for d in cluster_pair_data.values() if d.get('fallback'))
    n_trained = len(cluster_pair_data) - n_fallback
    print(f"\nCluster-pair Morse parameters ({len(cluster_pair_data)} pairs: "
          f"{n_trained} trained, {n_fallback} fallback):")
    for cp_name in sorted(cluster_pair_data.keys()):
        d = cluster_pair_data[cp_name]
        tag = " [FALLBACK]" if d.get('fallback') else ""
        max_d = d.get('max_edge_dist', 0)
        tc = d.get('training_cutoff', 0)
        print(f"  {cp_name:>22}: D_e={d['D_e']:.4f}+/-{d['D_e_std']:.4f} eV, "
              f"alpha={d['alpha']:.4f}+/-{d['alpha_std']:.4f} /A, "
              f"r0={d['r0']:.4f}+/-{d['r0_std']:.4f} A  "
              f"({d['n']} edges, max={max_d:.2f}A, cutoff={tc:.2f}A, "
              f"from {d['base']}){tag}")

    # Diagnose fallback pairs
    if n_fallback > 0:
        print(f"\n  -- Fallback Diagnosis ({n_fallback} pairs) --")
        organic_bases = {'C', 'H', 'O'}
        expected, surprising = [], []
        for cp_name, d in sorted(cluster_pair_data.items()):
            if not d.get('fallback'):
                continue
            ct1, ct2 = cp_name.split('-')
            b1, b2 = base_element(ct1), base_element(ct2)
            inorg1, inorg2 = b1 not in organic_bases, b2 not in organic_bases
            if (inorg1 != inorg2) or (inorg1 and inorg2 and b1 == b2):
                expected.append(f"    {cp_name:>30} ({d['base']})")
            else:
                surprising.append(f"    {cp_name:>30} ({d['base']})")
        if expected:
            print(f"  Expected (0 edges OK): {len(expected)}")
            for line in expected:
                print(line)
        if surprising:
            print(f"  Surprising (may need investigation): {len(surprising)}")
            for line in surprising:
                print(line)

    # Write output files
    os.makedirs(output_dir, exist_ok=True)

    # Atom type mapping
    mapping_path = os.path.join(output_dir, 'atom_type_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write(f"# Atom type mapping\n")
        f.write(f"# atom_index(1-based)  lammps_type  atom_type  original_element\n")
        f.write(f"# LAMMPS types: {', '.join(f'{e}={i+1}' for i, e in enumerate(cluster_elements))}\n")
        for i, (ct, orig) in enumerate(zip(atom_types, symbols)):
            lammps_type = cluster_elements.index(ct) + 1
            f.write(f"{i+1}  {lammps_type}  {ct}  {orig}\n")
    print(f"Saved: {mapping_path}")

    if original_data and os.path.exists(original_data):
        from .generate_lammps_files import (
            remap_data_file, write_lammps_input, add_missing_pairs,
        )
        params = add_missing_pairs(cluster_pair_data, cluster_elements)
        has_bonds = has_bonds_in_data(original_data)
        print(f"\nLAMPS export: original_data={original_data}, has_bonds={has_bonds}")

        remap_data_file(original_data, mapping_path,
                        os.path.join(output_dir, 'gnn_morse.data'),
                        cluster_elements)
        write_lammps_input(params, os.path.join(output_dir, 'gnn_morse.in'),
                           cluster_elements, temp=temp, has_bonds=has_bonds)

    return atom_types, type_legend, cluster_pair_data, cluster_elements
