import os

import numpy as np

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR,
    canonical_pair, base_element,
    has_bonds_in_data, detect_atom_style,
    MASSES, CHARMM_LJ, ORGANIC_ELEMENTS,
)
from .labelling import classify_atoms


#  _build_cluster_pair_edges

def _build_cluster_pair_edges(edge_index, atom_types, per_edge_params,
                              pair_indices, pair_names, frozen_pairs,
                              edge_distances_angstrom=None, fallback_De=0.001):
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
    # (skip organic-organic pairs — those use CHARMM LJ instead)
    cluster_types = sorted(set(atom_types))
    for i, ct1 in enumerate(cluster_types):
        for ct2 in cluster_types[i:]:
            cp_name = canonical_pair(ct1, ct2)
            if cp_name in result:
                continue
            if _is_organic_pair(cp_name):
                continue
            bp = canonical_pair(base_element(ct1), base_element(ct2))
            if bp in frozen_set or bp not in base_pair_medians:
                continue
            med = base_pair_medians[bp]
            result[cp_name] = {
                'D_e': fallback_De, 'alpha': med['alpha'], 'r0': med['r0'],
                'D_e_std': 0.0, 'alpha_std': 0.0, 'r0_std': 0.0,
                'n': 0, 'base': bp,
                'fallback': True, 'max_edge_dist': 0.0,
            }

    return result


#  LAMMPS file generation

def add_missing_pairs(params, cluster_elements, fallback_De=0.001):
    n_added = 0
    for i, ct1 in enumerate(cluster_elements):
        for ct2 in cluster_elements[i:]:
            cp = canonical_pair(ct1, ct2)
            if cp in params:
                continue
            if _is_organic_pair(cp):
                continue  # organic-organic pairs use CHARMM LJ
            bp = canonical_pair(base_element(ct1), base_element(ct2))
            params[cp] = {
                'D_e': fallback_De, 'alpha': 1.0, 'r0': 5.0,
                'n': 0, 'base': bp, 'fallback': True,
            }
            n_added += 1
    if n_added:
        print(f"Added {n_added} fallback pairs (H-Se etc.)")
    return params


def remap_data_file(original_path, mapping_path, output_path, cluster_elements):
    mapping = np.loadtxt(mapping_path, dtype=str, comments='#')
    atom_to_new_type = {int(row[0]): int(row[1]) for row in mapping}

    num_new_types = len(cluster_elements)
    elem_to_type = {e: i+1 for i, e in enumerate(cluster_elements)}

    atom_style = detect_atom_style(original_path)
    type_col = 2 if atom_style == 'full' else 1
    min_cols = 7 if atom_style == 'full' else 6

    with open(original_path) as f:
        lines = f.readlines()

    atoms_start = None
    atoms_end = None
    for i, line in enumerate(lines):
        if line.strip() == 'Atoms' or line.strip().startswith('Atoms #'):
            atoms_start = i + 2
        elif atoms_start is not None and atoms_end is None:
            if line.strip() in ('Bonds', 'Velocities', 'Angles', 'Dihedrals',
                                'Impropers', 'Pair Coeffs', 'Bond Coeffs'):
                atoms_end = i
    if atoms_end is None:
        atoms_end = len(lines)

    with open(output_path, 'w') as f:
        i = 0
        while i < len(lines):
            line = lines[i]

            if 'atom types' in line:
                f.write(f"     {num_new_types}  atom types\n")
                i += 1
                continue

            if line.strip() == 'Masses':
                f.write(line)
                f.write('\n')
                for elem in cluster_elements:
                    tid = elem_to_type[elem]
                    base = base_element(elem)
                    mass = MASSES.get(base, 1.0)
                    f.write(f"     {tid}   {mass:.4f}  # {elem}\n")
                i += 2
                while i < len(lines) and lines[i].strip():
                    i += 1
                continue

            if atoms_start is not None and i >= atoms_start and i < atoms_end:
                tokens = line.split()
                if len(tokens) >= min_cols and tokens[0].isdigit():
                    atom_id = int(tokens[0])
                    if atom_id in atom_to_new_type:
                        tokens[type_col] = str(atom_to_new_type[atom_id])
                    f.write(' '.join(tokens) + '\n')
                    i += 1
                    continue

            f.write(line)
            i += 1

    print(f"Wrote {output_path} ({num_new_types} atom types, {len(atom_to_new_type)} atoms remapped, "
          f"atom_style={atom_style})")


def _is_organic_pair(cp_name):
    e1, e2 = cp_name.split('-')
    b1, b2 = base_element(e1), base_element(e2)
    return b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS


def _write_pair_style_and_coeffs(f, params, cluster_elements, has_bonds,
                                 inorganic_elements=None):
    elem_to_type = {e: i+1 for i, e in enumerate(cluster_elements)}

    # Check if any organic elements exist (need hybrid for CHARMM LJ)
    has_organic = any(base_element(ce) in ORGANIC_ELEMENTS for ce in cluster_elements)

    # Max morse cutoff from training cutoffs
    max_morse_cutoff = 7.5
    for cp_name, data in params.items():
        if _is_organic_pair(cp_name):
            continue  # organic pairs use LJ, not Morse
        pair_cut = data.get('training_cutoff', 7.5)
        max_morse_cutoff = max(max_morse_cutoff, pair_cut)

    lj_cutoff = 4.0

    if has_organic:
        f.write(f"pair_style  hybrid/overlay morse {max_morse_cutoff:.1f} lj/cut {lj_cutoff:.1f}\n")
    else:
        f.write(f"pair_style  morse {max_morse_cutoff:.1f}\n")
    f.write("pair_modify shift yes\n")
    f.write("\n")

    # Morse pairs (inorganic-inorganic and inorganic-ligand)
    f.write("# -- MORSE PAIRS (GNN-fitted) --\n")
    f.write("#                         D_e(eV)      alpha(1/A)   r0(A)        cutoff(A)\n")

    for cp_name in sorted(params.keys()):
        if _is_organic_pair(cp_name):
            continue  # handled below as LJ
        data = params[cp_name]
        e1, e2 = cp_name.split('-')
        t1, t2 = elem_to_type.get(e1), elem_to_type.get(e2)
        if t1 is None or t2 is None:
            continue
        cutoff = data.get('training_cutoff', 7.5)
        fb = " [fallback]" if data.get('fallback') else ""
        if has_organic:
            f.write(f"pair_coeff  {t1} {t2} morse "
                    f"{data['D_e']:.8f}  {data['alpha']:.8f}  {data['r0']:.8f}  {cutoff:.2f}"
                    f"  # {cp_name}{fb}\n")
        else:
            f.write(f"pair_coeff  {t1} {t2} "
                    f"{data['D_e']:.8f}  {data['alpha']:.8f}  {data['r0']:.8f}  {cutoff:.2f}"
                    f"  # {cp_name}{fb}\n")

    # CHARMM LJ pairs (organic-organic only)
    if has_organic:
        f.write("\n# -- CHARMM LJ PAIRS (organic-organic, frozen) --\n")
        f.write("#                         epsilon(eV)  sigma(A)\n")
        written_lj = set()
        for i, ce1 in enumerate(cluster_elements):
            for ce2 in cluster_elements[i:]:
                cp = canonical_pair(ce1, ce2)
                if not _is_organic_pair(cp):
                    continue
                if cp in written_lj:
                    continue
                b1, b2 = base_element(ce1), base_element(ce2)
                bp = canonical_pair(b1, b2)
                lj = CHARMM_LJ.get(bp)
                if lj is None:
                    continue
                t1, t2 = elem_to_type[ce1], elem_to_type[ce2]
                f.write(f"pair_coeff  {t1} {t2} lj/cut "
                        f"{lj['epsilon']:.6f}  {lj['sigma']:.4f}"
                        f"  # {cp} (CHARMM)\n")
                written_lj.add(cp)


def write_lammps_input(params, filepath, cluster_elements, data_file='gnn_morse.data',
                       temp=300, nsteps=8707, has_bonds=True,
                       inorganic_elements=None):
    elem_names = [base_element(ce) for ce in cluster_elements]

    with open(filepath, 'w') as f:
        f.write(f"# LAMMPS Input: GNN-Morse fitted potential\n")
        f.write(f"# {len(cluster_elements)} atom types: "
                f"{', '.join(f'{e}={i+1}' for i, e in enumerate(cluster_elements))}\n\n")
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
            f.write("special_bonds    charmm\n")
        f.write(f"read_data       {data_file}\n")
        if has_bonds:
            f.write("special_bonds    charmm\n")
        f.write("\n")

        _write_pair_style_and_coeffs(f, params, cluster_elements, has_bonds,
                                     inorganic_elements=inorganic_elements)

        f.write(f"""
neighbor         2.0 bin
neigh_modify     delay 0 every 1 check yes
timestep         0.001

thermo           100
thermo_style     custom step temp pe ke etotal press
log              gnn_morse.log

dump             1 all custom 1 gnn_morse.xyz element xu yu zu vx vy vz fx fy fz
dump_modify      1 element {' '.join(elem_names)}
dump_modify      1 sort id

velocity         all create {temp} 12345
fix              1 all nve
fix              2 all temp/csvr {temp}.0 {temp}.0 0.1 54321

run              {nsteps}
""")
    print(f"Wrote {filepath}")


#  export_lammps

def export_lammps(model, dataset, config, output_dir,
                  original_data=None, temp=300, atom_detail='medium',
                  fallback_De=0.001):
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

    # Get cutoffs
    cutoffs_bohr = getattr(dataset, 'cutoffs_bohr', None)
    if cutoffs_bohr is None:
        gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
        cutoffs_bohr = {pair: cut * ANGSTROM_TO_BOHR
                        for pair, cut in gap_cuts.items()}

    # Atom classification
    vq_result = model_cpu.get_atom_types(graph)

    if vq_result is not None:
        # VQ active: use physical labelling
        from .labelling import compute_physical_labels
        _, vq_type_names = vq_result
        physical_labels, label_info = compute_physical_labels(
            symbols, dataset.frame_data[0][1], vq_type_names,
            inorganic, cutoffs_bohr,
            frame_data=dataset.frame_data, stride=10)
        atom_labels = physical_labels

        # Build type_legend from label_info
        type_legend = {}
        for label, info in label_info.items():
            base = label.partition('_')[0]
            if base in set(inorganic) and '_' in label:
                desc = (f"{base}, {label} (n={info['count']}, "
                        f"CN={info['mean_cn']:.1f}, "
                        f"ss={info['mean_surface_score']:.2f})")
            else:
                desc = f"{label} ({info['count']} atoms)"
            type_legend[label] = {'count': info['count'], 'description': desc}

        active_codes = model_cpu.vq.get_active_codes()
        active_str = ', '.join(f"{e}={n}" for e, n in active_codes.items()
                               if model_cpu.vq.n_codes_per_elem[e] > 1)
        print(f"\nAtom classification (VQ + physical labels, active: {active_str}):")
    else:
        # No VQ: fallback to CN-based classification
        atom_labels, type_legend, _ = classify_atoms(
            symbols, dataset.frame_data, cutoffs_bohr, inorganic,
            stride=10, atom_detail=atom_detail)

    atom_types = atom_labels

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
    edge_index_np = graph.edge_index.numpy()

    pair_indices = graph.pair_indices.numpy()
    edge_dists_ang = graph.distances.numpy() * BOHR_TO_ANGSTROM
    cluster_pair_data = _build_cluster_pair_edges(
        edge_index_np, atom_types, per_edge_params,
        pair_indices, dataset.pair_names, frozen_pairs,
        edge_distances_angstrom=edge_dists_ang, fallback_De=fallback_De)

    # Attach cutoffs for LAMMPS pair_style morse
    cutoffs_ang = getattr(dataset, 'gap_cuts_angstrom', None) or {}

    print(f"\nMorse cutoffs for LAMMPS (from training):")
    for cp_name, d in cluster_pair_data.items():
        bp = d['base']
        if bp in cutoffs_ang:
            d['training_cutoff'] = cutoffs_ang[bp]
        else:
            d['training_cutoff'] = 7.5
            print(f"  WARNING: {bp} not in training cutoffs, using 7.5 A fallback")

    # Print summary per base pair
    bp_cutoffs = {}
    for cp_name, d in cluster_pair_data.items():
        bp = d['base']
        tc = d.get('training_cutoff', 7.5)
        bp_cutoffs[bp] = max(bp_cutoffs.get(bp, 0), tc)
    for bp in sorted(bp_cutoffs):
        print(f"  {bp}: {bp_cutoffs[bp]:.2f} A")

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
        organic_bases = set(base_element(ce) for ce in cluster_elements) - set(inorganic)
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
        params = add_missing_pairs(cluster_pair_data, cluster_elements, fallback_De=fallback_De)
        has_bonds = has_bonds_in_data(original_data)
        print(f"\nLAMPS export: original_data={original_data}, has_bonds={has_bonds}")

        remap_data_file(original_data, mapping_path,
                        os.path.join(output_dir, 'gnn_morse.data'),
                        cluster_elements)
        write_lammps_input(params, os.path.join(output_dir, 'gnn_morse.in'),
                           cluster_elements, temp=temp, has_bonds=has_bonds,
                           inorganic_elements=inorganic)

    return atom_types, type_legend, cluster_pair_data, cluster_elements
