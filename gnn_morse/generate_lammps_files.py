#!/usr/bin/env python3
import os
import numpy as np

from .utils import (
    base_element, canonical_pair, MASSES, FROZEN_LJ, BASE_PAIR_CUTOFFS,
    detect_atom_style, has_bonds_in_data,
)


def add_missing_pairs(params, cluster_elements):
    frozen_bases = set(FROZEN_LJ.keys())
    n_added = 0
    for i, ct1 in enumerate(cluster_elements):
        for ct2 in cluster_elements[i:]:
            cp = canonical_pair(ct1, ct2)
            if cp in params:
                continue
            bp = canonical_pair(base_element(ct1), base_element(ct2))
            if bp in frozen_bases:
                continue
            params[cp] = {
                'D_e': 0.001, 'alpha': 1.0, 'r0': 5.0,
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



def _write_pair_style_and_coeffs(f, params, cluster_elements, has_bonds):
    elem_to_type = {e: i+1 for i, e in enumerate(cluster_elements)}

    # Max morse cutoff from training cutoffs
    max_morse_cutoff = 7.5
    for data in params.values():
        train_cut = data.get('training_cutoff', 0.0)
        pair_cut = train_cut if train_cut > 0 else BASE_PAIR_CUTOFFS.get(data['base'], 7.5)
        max_morse_cutoff = max(max_morse_cutoff, pair_cut)

    organic_elements = {'C', 'H', 'O'}
    has_organic = any(base_element(ce) in organic_elements for ce in cluster_elements)
    use_lj = has_bonds and has_organic

    if use_lj:
        f.write(f"pair_style  hybrid/overlay morse/smooth/linear {max_morse_cutoff:.1f} lj/cut 10.0\n")
    else:
        f.write(f"pair_style  morse/smooth/linear {max_morse_cutoff:.1f}\n")
    f.write("\n")

    # Morse pairs
    f.write("# ── GNN-TRAINED MORSE PAIRS ──\n")
    f.write("#                          D_e(eV)      alpha(1/A)   r0(A)        cutoff(A)\n")

    for cp_name in sorted(params.keys()):
        data = params[cp_name]
        e1, e2 = cp_name.split('-')
        t1, t2 = elem_to_type.get(e1), elem_to_type.get(e2)
        if t1 is None or t2 is None:
            continue
        train_cut = data.get('training_cutoff', 0.0)
        cutoff = train_cut if train_cut > 0 else BASE_PAIR_CUTOFFS.get(data['base'], 7.5)
        fb = " [fallback]" if data.get('fallback') else ""
        style_prefix = "morse/smooth/linear " if use_lj else ""
        f.write(f"pair_coeff  {t1} {t2} {style_prefix}"
                f"{data['D_e']:.8f}  {data['alpha']:.8f}  {data['r0']:.8f}  {cutoff:.2f}"
                f"  # {cp_name}{fb}\n")

    # Frozen organic LJ pairs
    if use_lj:
        f.write("\n# ── FROZEN ORGANIC LJ PAIRS (CHARMM) ──\n")
        base_to_clusters = {}
        for ce in cluster_elements:
            base_to_clusters.setdefault(base_element(ce), []).append(ce)

        for pn in sorted(FROZEN_LJ.keys()):
            vals = FROZEN_LJ[pn]
            base_e1, base_e2 = pn.split('-')
            for v1 in base_to_clusters.get(base_e1, []):
                for v2 in base_to_clusters.get(base_e2, []):
                    t1, t2 = elem_to_type[v1], elem_to_type[v2]
                    if t1 > t2:
                        t1, t2 = t2, t1
                    f.write(f"pair_coeff  {t1} {t2} lj/cut "
                            f"{vals['eps']:.8f}  {vals['sigma']:.5f}"
                            f"  # {canonical_pair(v1, v2)} ({pn})\n")


def write_lammps_input(params, filepath, cluster_elements, data_file='gnn_morse.data',
                       temp=300, nsteps=8707, has_bonds=True):
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

        _write_pair_style_and_coeffs(f, params, cluster_elements, has_bonds)

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



