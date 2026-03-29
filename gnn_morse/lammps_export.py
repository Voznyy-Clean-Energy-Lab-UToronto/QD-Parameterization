import os
import numpy as np
from datetime import datetime

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, canonical_pair, base_element, MASSES,
    has_bonds_in_data,
)


def _describe_vq_types(atom_types, symbols, edge_index_np, inorganic_elements):
    from collections import Counter

    inorganic_set = set(inorganic_elements)
    n_atoms = len(symbols)
    src, tgt = edge_index_np[0], edge_index_np[1]

    # Per-atom neighbor analysis
    cross_coord = np.zeros(n_atoms, dtype=int)       # different-element inorganic neighbors
    total_inorg_coord = np.zeros(n_atoms, dtype=int)  # all inorganic neighbors
    cross_neighbor_elem = [Counter() for _ in range(n_atoms)]
    has_ligand = np.zeros(n_atoms, dtype=bool)

    for s, t in zip(src, tgt):
        src_elem = symbols[s]
        tgt_elem = symbols[t]
        if src_elem in inorganic_set:
            if tgt_elem in inorganic_set:
                total_inorg_coord[s] += 1
                if tgt_elem != src_elem:
                    cross_coord[s] += 1
                    cross_neighbor_elem[s][tgt_elem] += 1
            else:
                has_ligand[s] = True

    # Per-element max cross-species coord (core = atoms matching this max)
    elem_max_cross = {}
    for elem in inorganic_set:
        indices = [i for i, s in enumerate(symbols) if s == elem]
        if indices:
            elem_max_cross[elem] = int(np.max(cross_coord[indices]))

    unique_types = sorted(set(atom_types))
    label_map = {}

    for tname in unique_types:
        elem = tname.partition('_')[0]

        # Non-inorganic: C_0 -> C, H_0 -> H, but O with multiple types -> O_0, O_1
        if elem not in inorganic_set:
            elem_types = [t for t in unique_types if t.partition('_')[0] == elem]
            if len(elem_types) == 1:
                label_map[tname] = elem
            else:
                label_map[tname] = tname
            continue

        indices = [i for i, t in enumerate(atom_types) if t == tname]
        median_cross = int(np.round(np.median(cross_coord[indices])))
        median_total = int(np.round(np.median(total_inorg_coord[indices])))
        frac_ligand = float(np.mean(has_ligand[indices]))
        max_cross = elem_max_cross.get(elem, 4)

        # Core = matches max cross-species coordination, surface = lower
        location = "core" if median_cross >= max_cross else "surf"

        # Primary cross-species neighbor element
        all_neighbors = Counter()
        for i in indices:
            all_neighbors.update(cross_neighbor_elem[i])
        primary_neighbor = all_neighbors.most_common(1)[0][0] if all_neighbors else ""

        coord_tag = f"_{median_cross}{primary_neighbor}" if primary_neighbor else ""
        lig_tag = "_lig" if frac_ligand >= 0.5 else ""
        label_map[tname] = f"{elem}_{location}{coord_tag}{lig_tag}"

    # Resolve duplicates: append _a, _b, _c letter suffixes
    label_values = list(label_map.values())
    counts = Counter(label_values)
    if any(c > 1 for c in counts.values()):
        seen = Counter()
        for tname in unique_types:
            label = label_map[tname]
            if counts[label] > 1:
                seen[label] += 1
                label_map[tname] = f"{label}_{chr(96 + seen[label])}"

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
    edge_index_np = graph.edge_index.numpy()

    # ── Atom classification ──
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
        print(f"\n  -- Fallback Diagnosis ({n_fallback} pairs) --")
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
