import os
from collections import Counter
import numpy as np

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR,
    canonical_pair, base_element, MASSES, CHARMM_LJ, ORGANIC_ELEMENTS,
)


def _remap_data_file(original_path, mapping_path, output_path, cluster_elements):
    mapping = np.loadtxt(mapping_path, dtype=str, comments='#')
    atom_to_new_type = {int(row[0]): int(row[1]) for row in mapping}
    num_new_types = len(cluster_elements)
    elem_to_type = {e: i+1 for i, e in enumerate(cluster_elements)}

    with open(original_path) as f:
        lines = f.readlines()

    # find Atoms section
    atoms_start, atoms_end = None, None
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
                    mass = MASSES.get(base_element(elem), 1.0)
                    f.write(f"     {tid}   {mass:.4f}  # {elem}\n")
                i += 2
                while i < len(lines) and lines[i].strip():
                    i += 1
                continue

            # remap atom types in Atoms section
            if atoms_start is not None and atoms_start <= i < atoms_end:
                tokens = line.split()
                if len(tokens) >= 7 and tokens[0].isdigit():
                    atom_id = int(tokens[0])
                    if atom_id in atom_to_new_type:
                        tokens[2] = str(atom_to_new_type[atom_id])
                    f.write(' '.join(tokens) + '\n')
                    i += 1
                    continue

            f.write(line)
            i += 1

    print(f"  Wrote {output_path} ({num_new_types} types)")


def _write_lammps_input(params, filepath, cluster_elements, data_file='gnn_morse.data',
                        temp=300, nsteps=8707):
    elem_to_type = {e: i+1 for i, e in enumerate(cluster_elements)}
    elem_names = [base_element(ce) for ce in cluster_elements]
    has_organic = any(base_element(ce) in ORGANIC_ELEMENTS for ce in cluster_elements)

    # check for bonds in data file
    has_bonds = False
    data_path = os.path.join(os.path.dirname(filepath), data_file)
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                if line.strip().endswith('bonds') and not line.strip().startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) > 0:
                        has_bonds = True
                        break

    max_morse_cutoff = max(
        (d.get('training_cutoff', 7.5) for cp, d in params.items()
         if not (base_element(cp.split('-')[0]) in ORGANIC_ELEMENTS and
                 base_element(cp.split('-')[1]) in ORGANIC_ELEMENTS)),
        default=7.5)

    with open(filepath, 'w') as f:
        f.write(f"# LAMMPS Input\n")
        f.write(f"# {len(cluster_elements)} atom types: "
                f"{', '.join(f'{e}={i+1}' for i, e in enumerate(cluster_elements))}\n\n")
        f.write("units           metal\n")

        if has_bonds:
            f.write("atom_style      full\nimproper_style  harmonic\n")
            f.write("bond_style      harmonic\nangle_style     harmonic\n")
        else:
            f.write("atom_style      charge\n")
        f.write("boundary        f f f\n\n")

        if has_bonds:
            f.write("special_bonds    charmm\n")
        f.write(f"read_data       {data_file}\n")
        if has_bonds:
            f.write("special_bonds    charmm\n")
        f.write("\n")

        # pair style
        if has_organic:
            f.write(f"pair_style  hybrid/overlay morse {max_morse_cutoff:.1f} lj/cut 4.0\n")
        else:
            f.write(f"pair_style  morse {max_morse_cutoff:.1f}\n")
        f.write("pair_modify     shift yes\n\n")

        # Morse pairs
        f.write("# -- MORSE PAIRS --\n")
        for cp in sorted(params.keys()):
            e1, e2 = cp.split('-')
            if base_element(e1) in ORGANIC_ELEMENTS and base_element(e2) in ORGANIC_ELEMENTS:
                continue
            d = params[cp]
            t1, t2 = elem_to_type.get(e1), elem_to_type.get(e2)
            if t1 is None or t2 is None:
                continue
            cutoff = d.get('training_cutoff', 7.5)
            fb = " [fallback]" if d.get('fallback') else ""
            style = "morse " if has_organic else ""
            f.write(f"pair_coeff  {t1} {t2} {style}"
                    f"{d['D_e']:.8f}  {d['alpha']:.8f}  {d['r0']:.8f}  {cutoff:.2f}"
                    f"  # {cp}{fb}\n")

        # CHARMM LJ pairs
        if has_organic:
            f.write("\n# -- CHARMM LJ PAIRS --\n")
            written = set()
            for i, ce1 in enumerate(cluster_elements):
                for ce2 in cluster_elements[i:]:
                    cp = canonical_pair(ce1, ce2)
                    b1, b2 = base_element(ce1), base_element(ce2)
                    if not (b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS):
                        continue
                    if cp in written:
                        continue
                    bp = canonical_pair(b1, b2)
                    lj = CHARMM_LJ.get(bp)
                    if lj is None:
                        continue
                    t1, t2 = elem_to_type[ce1], elem_to_type[ce2]
                    f.write(f"pair_coeff  {t1} {t2} lj/cut "
                            f"{lj['epsilon']:.6f}  {lj['sigma']:.4f}  # {cp}\n")
                    written.add(cp)

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
    print(f"  Wrote {filepath}")


def export_lammps(model, dataset, config, atom_type_names, unique_types,
                  cpair_names, cpair_to_base, output_dir, temp=300):
    print(f"\n{'='*60}\nLAMMPS EXPORT\n{'='*60}")

    symbols = dataset.frame_data[0][0]
    fallback_De = config.get('fallback_De', 0.001)
    cluster_elements = unique_types

    tc = Counter(atom_type_names)
    print(f"\nAtom types ({len(cluster_elements)}):")
    for t in cluster_elements:
        print(f"  {t:>15}: {tc[t]:>3} atoms")

    # get trained parameters
    table_params = model.get_type_pair_params()
    pair_data = {}
    for cpair_name, data in table_params.items():
        t1, t2 = cpair_name.split('-')
        if t1 not in cluster_elements or t2 not in cluster_elements:
            continue
        bp = cpair_to_base.get(cpair_name, cpair_name)
        pair_data[cpair_name] = {
            'D_e': data['D_e'], 'alpha': data['alpha'], 'r0': data['r0'],
            'base': bp, 'fallback': data['D_e'] < 1e-4,
        }

    # per-subtype cutoffs, fallback to element-pair
    subtype_cuts = getattr(dataset, 'subtype_cutoffs_angstrom', {}) or {}
    elem_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
    for cp, d in pair_data.items():
        d['training_cutoff'] = subtype_cuts.get(cp, elem_cuts.get(d['base'], 7.5))

    # add fallback for missing pairs
    for i, ct1 in enumerate(cluster_elements):
        for ct2 in cluster_elements[i:]:
            cp = canonical_pair(ct1, ct2)
            b1, b2 = base_element(ct1), base_element(ct2)
            if cp in pair_data or (b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS):
                continue
            pair_data[cp] = {
                'D_e': fallback_De, 'alpha': 1.0, 'r0': 5.0,
                'base': canonical_pair(b1, b2), 'fallback': True,
                'training_cutoff': elem_cuts.get(canonical_pair(b1, b2), 7.5),
            }

    # print active parameters
    print(f"\nMorse parameters:")
    for cp in sorted(pair_data.keys()):
        d = pair_data[cp]
        b1, b2 = [base_element(x) for x in cp.split('-')]
        if (b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS) or d['D_e'] < 1e-4:
            continue
        print(f"  {cp:>30}: D_e={d['D_e']:.6f} eV, alpha={d['alpha']:.6f} /A, r0={d['r0']:.6f} A")

    # write files
    os.makedirs(output_dir, exist_ok=True)

    mapping_path = os.path.join(output_dir, 'atom_type_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write(f"# atom_index  lammps_type  atom_type  element\n")
        for i, (ct, orig) in enumerate(zip(atom_type_names, symbols)):
            lammps_type = cluster_elements.index(ct) + 1
            f.write(f"{i+1}  {lammps_type}  {ct}  {orig}\n")

    original_data = config.get('original_data')
    if original_data and os.path.exists(original_data):
        _remap_data_file(original_data, mapping_path,
                         os.path.join(output_dir, 'gnn_morse.data'),
                         cluster_elements)
        _write_lammps_input(pair_data, os.path.join(output_dir, 'gnn_morse.in'),
                            cluster_elements, temp=temp)

    print(f"LAMMPS files: {output_dir}")
