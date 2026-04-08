import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .utils import BOHR_TO_ANGSTROM, canonical_pair, ORGANIC_ELEMENTS


#  Helpers

def _compute_cross_species_cn(symbols, positions_bohr, inorganic_elements,
                              cutoffs_bohr):
    #Labelling looks at cross-species CN. CN is not used for making thetypes, but is used in labelling.
    n_atoms = len(symbols)
    inorganic_set = set(inorganic_elements)
    elem_of = np.array(symbols)

    # Build per-pair cutoffs in Angstrom
    inorg_elems = sorted(inorganic_set & set(symbols))
    pair_cutoffs_ang = {}
    max_cutoff_ang = 0.0
    for ia, ea in enumerate(inorg_elems):
        for eb in inorg_elems[ia + 1:]:
            pk = canonical_pair(ea, eb)
            cut = cutoffs_bohr.get(pk)
            if cut is not None:
                cut_ang = cut * BOHR_TO_ANGSTROM
                pair_cutoffs_ang[pk] = cut_ang
                max_cutoff_ang = max(max_cutoff_ang, cut_ang)
    if max_cutoff_ang == 0.0:
        max_cutoff_ang = 5.0

    pos_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM
    atoms = Atoms(symbols=symbols, positions=pos_ang)
    i_list, j_list, d_list = neighbor_list('ijd', atoms, max_cutoff_ang,
                                            self_interaction=False)

    cn = np.zeros(n_atoms, dtype=np.float64)
    for idx in range(len(i_list)):
        ai, aj = i_list[idx], j_list[idx]
        ei, ej = elem_of[ai], elem_of[aj]
        if ei == ej:
            continue
        if ei not in inorganic_set or ej not in inorganic_set:
            continue
        pk = canonical_pair(ei, ej)
        cut_ang = pair_cutoffs_ang.get(pk)
        if cut_ang is None:
            continue
        if d_list[idx] <= cut_ang:
            cn[ai] += 1

    return cn


def _compute_surface_score(symbols, positions_bohr, inorganic_elements,
                           cutoffs_bohr):
    #Compute surface score per atom: ||mean(unit_vectors_to_inorganic_neighbors)||.
    #0 = symmetric (core), 1 = all neighbors on one side (edge/vertex). This surface score idea is totally made up, but is convenient. Labelling in general is not the 
    #most robust, but I just needed it to be consistent to be able to evaluate consistency and read potentials.png. More work needs to be done. 
    #We need to find out what physical properties actually matter most, because we already it isnt CN.
    n_atoms = len(symbols)
    inorganic_set = set(inorganic_elements)
    elem_of = np.array(symbols)

    inorg_elems = sorted(inorganic_set & set(symbols))
    pair_cutoffs_ang = {}
    max_cutoff_ang = 0.0
    for ia, ea in enumerate(inorg_elems):
        for eb in inorg_elems[ia + 1:]:
            pk = canonical_pair(ea, eb)
            cut = cutoffs_bohr.get(pk)
            if cut is not None:
                cut_ang = cut * BOHR_TO_ANGSTROM
                pair_cutoffs_ang[pk] = cut_ang
                max_cutoff_ang = max(max_cutoff_ang, cut_ang)
    if max_cutoff_ang == 0.0:
        max_cutoff_ang = 5.0

    pos_ang = np.asarray(positions_bohr) * BOHR_TO_ANGSTROM
    atoms = Atoms(symbols=symbols, positions=pos_ang)
    i_list, j_list, d_list = neighbor_list('ijd', atoms, max_cutoff_ang,
                                            self_interaction=False)

    # Accumulate unit vectors from each atom to its inorganic neighbors
    vec_sum = np.zeros((n_atoms, 3), dtype=np.float64)
    neighbor_count = np.zeros(n_atoms, dtype=np.float64)

    for idx in range(len(i_list)):
        ai, aj = i_list[idx], j_list[idx]
        ei, ej = elem_of[ai], elem_of[aj]
        if ei not in inorganic_set or ej not in inorganic_set:
            continue
        pk = canonical_pair(ei, ej)
        cut_ang = pair_cutoffs_ang.get(pk)
        if cut_ang is None:
            continue
        if d_list[idx] > cut_ang:
            continue
        # Unit vector from ai to aj
        diff = pos_ang[aj] - pos_ang[ai]
        dist = d_list[idx]
        if dist < 1e-8:
            continue
        vec_sum[ai] += diff / dist
        neighbor_count[ai] += 1

    scores = np.zeros(n_atoms, dtype=np.float64)
    has_neighbors = neighbor_count > 0
    scores[has_neighbors] = (np.linalg.norm(vec_sum[has_neighbors], axis=1)
                             / neighbor_count[has_neighbors])
    return scores


def _assign_position_tags(group_surface_scores):
    n = len(group_surface_scores)
    if n <= 1:
        return {code: '' for code in group_surface_scores}

    # Sort by ascending surface score (lowest = most core-like)
    sorted_codes = sorted(group_surface_scores, key=group_surface_scores.get)

    if n == 2:
        tags = ['core', 'surf']
    else:
        # 3+: core, surf, edge (if exactly 3). For 4+, intermediate ones are surf.
        tags = ['core'] + ['surf'] * (n - 2) + ['edge']

    return {code: tag for code, tag in zip(sorted_codes, tags)}


#  Main API

def compute_physical_labels(symbols, positions_bohr, vq_type_names,
                            inorganic_elements, cutoffs_bohr,
                            frame_data=None, stride=10):
    n_atoms = len(symbols)
    inorganic_set = set(inorganic_elements)

    # Determine which frames to use
    if frame_data is not None and len(frame_data) > 0:
        frame_indices = range(0, len(frame_data), max(1, stride))
    else:
        frame_data = [(symbols, positions_bohr)]
        frame_indices = range(1)

    # Accumulate CN and surface score across frames
    cn_accum = np.zeros(n_atoms, dtype=np.float64)
    ss_accum = np.zeros(n_atoms, dtype=np.float64)
    n_frames = 0

    for fi in frame_indices:
        syms_f, pos_f = frame_data[fi]
        cn_accum += _compute_cross_species_cn(
            syms_f, pos_f, inorganic_elements, cutoffs_bohr)
        ss_accum += _compute_surface_score(
            syms_f, pos_f, inorganic_elements, cutoffs_bohr)
        n_frames += 1

    if n_frames > 0:
        avg_cn = cn_accum / n_frames
        avg_ss = ss_accum / n_frames
    else:
        avg_cn = cn_accum
        avg_ss = ss_accum

    # Identify inorganic VQ groups (codes with underscores for multi-code elements)
    # Group atoms by their VQ code
    code_to_indices = {}
    for i, code in enumerate(vq_type_names):
        code_to_indices.setdefault(code, []).append(i)

    # Separate inorganic codes by base element
    elem_codes = {}  # elem -> list of codes
    for code in sorted(code_to_indices.keys()):
        base = code.partition('_')[0]
        if base in inorganic_set and '_' in code:
            elem_codes.setdefault(base, []).append(code)

    # Build label mapping: code -> physical label
    code_to_label = {}

    for elem, codes in elem_codes.items():
        # Determine partner element (cross-species)
        partners = sorted(inorganic_set - {elem})
        partner = partners[0] if partners else elem

        # Compute group-level stats
        group_cn = {}
        group_ss = {}
        for code in codes:
            idx = code_to_indices[code]
            group_cn[code] = float(np.mean(avg_cn[idx]))
            group_ss[code] = float(np.mean(avg_ss[idx]))

        # Assign position tags
        pos_tags = _assign_position_tags(group_ss)

        # Build label strings
        labels_for_elem = {}
        for code in codes:
            cn_int = int(round(group_cn[code]))
            tag = pos_tags[code]
            if tag:
                label = f"{elem}_{tag}_{partner}{cn_int}"
            else:
                label = f"{elem}_{partner}{cn_int}"
            labels_for_elem[code] = label

        # Disambiguate collisions
        label_counts = {}
        for code, label in labels_for_elem.items():
            label_counts.setdefault(label, []).append(code)

        for label, colliding_codes in label_counts.items():
            if len(colliding_codes) > 1:
                # Sort by surface_score ascending, add letter suffix
                sorted_by_ss = sorted(colliding_codes, key=lambda c: group_ss[c])
                for i, code in enumerate(sorted_by_ss):
                    labels_for_elem[code] = label + chr(ord('a') + i)

        code_to_label.update(labels_for_elem)

    # Assign labels to all atoms
    physical_labels = []
    for i in range(n_atoms):
        code = vq_type_names[i]
        base = code.partition('_')[0]
        if base not in inorganic_set or '_' not in code:
            # Organic or single-code inorganic: use bare element name
            physical_labels.append(base)
        else:
            physical_labels.append(code_to_label.get(code, code))

    # Build label_info
    label_info = {}
    for label in sorted(set(physical_labels)):
        idx = [i for i, l in enumerate(physical_labels) if l == label]
        label_info[label] = {
            'count': len(idx),
            'mean_cn': float(np.mean(avg_cn[idx])),
            'mean_surface_score': float(np.mean(avg_ss[idx])),
        }

    # Print summary
    print(f"\n  Physical labels ({n_frames} frames, stride={stride}):")
    for label in sorted(label_info.keys()):
        info = label_info[label]
        base = label.partition('_')[0]
        if base in inorganic_set and '_' in label:
            print(f"    {label:>20}: n={info['count']:>3}, "
                  f"CN={info['mean_cn']:.1f}, "
                  f"surface_score={info['mean_surface_score']:.2f}")
        else:
            print(f"    {label:>20}: n={info['count']:>3}")

    return physical_labels, label_info


#  classify_atoms

def classify_atoms(symbols, frame_data, cutoffs_bohr, inorganic_elements,
                   vq_atom_types=None, stride=10,
                   atom_detail='medium'):
    inorganic_set = set(inorganic_elements)
    n_atoms = len(symbols)
    symbols_arr = np.array(symbols)

    # Identify inorganic atoms by element
    inorganic_atom_indices = {}
    for elem in sorted(inorganic_set):
        mask = symbols_arr == elem
        if mask.any():
            inorganic_atom_indices[elem] = np.where(mask)[0]
    inorg_elems = sorted(inorganic_atom_indices.keys())

    # Accumulate cross-species CN per atom across strided frames
    cn_accum = np.zeros(n_atoms, dtype=np.float64)
    n_frames_used = 0

    frame_indices = range(0, len(frame_data), max(1, stride))
    for fi in frame_indices:
        syms_f, pos_f = frame_data[fi]
        cn_accum += _compute_cross_species_cn(
            syms_f, pos_f, inorganic_elements, cutoffs_bohr)
        n_frames_used += 1

    if n_frames_used > 0:
        avg_cn = cn_accum / n_frames_used
    else:
        avg_cn = cn_accum

    # Classify each inorganic element into groups
    atom_group = np.full(n_atoms, -1, dtype=int)  # -1 = ligand
    group_names = {}  # {(elem, group_id): label}

    for elem in inorg_elems:
        idx = inorganic_atom_indices[elem]
        cns = avg_cn[idx]

        # Determine number of groups
        if atom_detail == 'base':
            # No splitting
            atom_group[idx] = 0
            group_names[(elem, 0)] = elem
            continue

        if vq_atom_types is not None:
            elem_codes = sorted(set(vq_atom_types[i] for i in idx))
            n_grp = len(elem_codes)
        else:
            n_grp = 2

        if n_grp <= 1 or len(np.unique(np.round(cns, 1))) <= 1:
            atom_group[idx] = 0
            group_names[(elem, 0)] = elem
            continue

        threshold = (cns.min() + cns.max()) / 2.0
        for i, ai in enumerate(idx):
            atom_group[ai] = 0 if cns[i] >= threshold else 1
        group_names[(elem, 0)] = f"{elem}_core"
        group_names[(elem, 1)] = f"{elem}_surf"

    # Build per-atom labels
    atom_labels = []
    for i in range(n_atoms):
        sym = symbols[i]
        if sym not in inorganic_set:
            atom_labels.append(sym)
        else:
            gid = atom_group[i]
            label = group_names.get((sym, gid), sym)
            atom_labels.append(label)

    # If VQ codes provided: use them directly as atom types
    if vq_atom_types is not None:
        atom_labels = list(vq_atom_types)

    # Build type legend
    type_legend = {}
    for label in sorted(set(atom_labels)):
        count = sum(1 for l in atom_labels if l == label)
        elem = label.partition('_')[0]
        member_idx = [i for i, l in enumerate(atom_labels) if l == label]
        mean_cn = float(np.mean(avg_cn[member_idx])) if member_idx else 0.0
        if elem in inorganic_set and '_' in label:
            desc = f"{elem}, {label} (n={count}, mean_CN={mean_cn:.1f})"
        else:
            desc = f"{label} ({count} atoms)"
        type_legend[label] = {'count': count, 'description': desc}

    # Print CN distribution summary
    print(f"\n  Cross-species CN distribution ({n_frames_used} frames, stride={stride}):")
    for elem in inorg_elems:
        idx = inorganic_atom_indices[elem]
        cns = avg_cn[idx]
        print(f"    {elem}: min={cns.min():.1f}, max={cns.max():.1f}, "
              f"mean={cns.mean():.1f}, median={np.median(cns):.1f}")
        for label in sorted(set(atom_labels[i] for i in idx)):
            label_idx = [i for i in idx if atom_labels[i] == label]
            label_cns = avg_cn[label_idx]
            print(f"      {label}: n={len(label_idx)}, "
                  f"CN={label_cns.min():.1f}-{label_cns.max():.1f} "
                  f"(mean={label_cns.mean():.1f})")

    return atom_labels, type_legend, avg_cn
