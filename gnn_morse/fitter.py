import os
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .data import DFTDataset, detect_gap_cutoff
from .models import MorseModel, assign_types, build_type_pair_infrastructure
from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    ORGANIC_ELEMENTS, FORCE_AU_TO_EV_ANG, canonical_pair, base_element,
)


def load_config(filepath):
    with open(filepath) as f:
        raw = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(filepath))
    config = {}
    config.update(raw.get('training', {}))
    config['datasets'] = raw.get('datasets', [])
    for ds in config['datasets']:
        if 'xyz' in ds and not os.path.isabs(ds['xyz']):
            ds['xyz'] = os.path.normpath(os.path.join(config_dir, ds['xyz']))
    config['knn_edges'] = raw['knn_edges']
    for key in ['original_data', 'reference_structure']:
        val = raw.get(key)
        if val and not os.path.isabs(val):
            val = os.path.normpath(os.path.join(config_dir, val))
        config[key] = val
    return config


def force_rmse(predicted, target, element_indices, core_mask=None):
    sq_err = (predicted - target) ** 2
    if core_mask is not None:
        mask = core_mask[element_indices].unsqueeze(1)
        return torch.sqrt((sq_err * mask).sum() / (mask.sum() * 3) + 1e-30)
    return torch.sqrt(sq_err.mean() + 1e-30)


def log(f, msg):
    f.write(msg + '\n')
    f.flush()


def main():
    parser = argparse.ArgumentParser(description='PMF init + Morse force matching')
    parser.add_argument('config', help='Path to config YAML')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    train(config, output_dir)


def train(config, output_dir):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join(output_dir, 'results')
    lammps_dir = os.path.join(output_dir, 'lammps')
    os.makedirs(results_dir, exist_ok=True)
    inorganic = config['knn_edges']['inorganic_elements']

    logfile = open(os.path.join(results_dir, 'training.log'), 'w')
    log(logfile, f"Device: {device}")

    #load data
    dataset = DFTDataset(config['datasets'], config['knn_edges'],
                         first_n_frames=config.get('first_n_frames'))
    dataset.build_graphs()

    #assign types from frame 0
    symbols = dataset.frame_data[0][0]
    pos_ang = np.asarray(dataset.frame_data[0][1]) * BOHR_TO_ANGSTROM
    n_atoms = len(symbols)

    atom_type_names, q6, org_cn = assign_types(symbols, pos_ang, inorganic)
    unique_types = sorted(set(atom_type_names))
    type_to_index = {t: i for i, t in enumerate(unique_types)}
    atom_type_indices = np.array([type_to_index[t] for t in atom_type_names])
    types_arr = np.array(atom_type_names)
    syms_arr = np.array(symbols)

    log(logfile, f"\nTypes ({len(unique_types)}):")
    for t, count in sorted(Counter(atom_type_names).items()):
        idx = [i for i, x in enumerate(atom_type_names) if x == t]
        log(logfile, f"  {t:>15}: {count:>3}  q6={q6[idx].mean():.3f}  org_cn={org_cn[idx].mean():.1f}")

    #type-pair infrastructure
    cpair_names, cpair_lookup, cpair_to_base = build_type_pair_infrastructure(unique_types)
    num_type_pairs = len(cpair_names)

    #per-subtype gap detection + edge filtering
    cpair_lookup_t = torch.tensor(cpair_lookup, dtype=torch.long)
    atom_type_t = torch.tensor(atom_type_indices, dtype=torch.long)
    dmat_bohr = cdist(np.asarray(dataset.frame_data[0][1]),
                      np.asarray(dataset.frame_data[0][1]))
    np.fill_diagonal(dmat_bohr, np.inf)
    dmat_ang = dmat_bohr * BOHR_TO_ANGSTROM

    subtype_cutoffs_bohr = {}
    log(logfile, f"\nPer-subtype cutoffs:")
    for i_t, t1 in enumerate(unique_types):
        idx1 = np.where(types_arr == t1)[0]
        for t2 in unique_types[i_t:]:
            idx2 = np.where(types_arr == t2)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            cp = canonical_pair(t1, t2)
            bp = cpair_to_base.get(cp, cp)
            if t1 == t2:
                d = dmat_bohr[np.ix_(idx1, idx2)].copy()
                np.fill_diagonal(d, np.inf)
            else:
                d = dmat_bohr[np.ix_(idx1, idx2)]
            cut = detect_gap_cutoff(d, config['knn_edges'].get('max_gap_neighbors', 20))
            elem_cut = dataset.cutoffs_bohr.get(bp, 100.0 * ANGSTROM_TO_BOHR)
            if not np.isinf(cut):
                cut = min(cut, elem_cut)
            if np.isinf(cut) or cut > 15.0 * ANGSTROM_TO_BOHR:
                continue
            subtype_cutoffs_bohr[cp] = cut
            log(logfile, f"  {cp:>25}: {cut * BOHR_TO_ANGSTROM:.2f} A")

    dataset.subtype_cutoffs_angstrom = {p: c * BOHR_TO_ANGSTROM for p, c in subtype_cutoffs_bohr.items()}

    # filter edges to per-subtype cutoffs
    n_removed, n_total = 0, 0
    for g in dataset.graphs:
        src, tgt = g.edge_index
        type_pair_idx = cpair_lookup_t[atom_type_t[src], atom_type_t[tgt]]

        keep = torch.ones(len(src), dtype=torch.bool)
        for ei in range(len(src)):
            cp = cpair_names[type_pair_idx[ei].item()]
            if cp not in subtype_cutoffs_bohr or g.distances[ei] > subtype_cutoffs_bohr[cp]:
                keep[ei] = False

        n_removed += (~keep).sum().item()
        n_total += len(src)
        g.edge_index = g.edge_index[:, keep]
        g.distances = g.distances[keep]
        g.edge_unit_vectors = g.edge_unit_vectors[keep]
        g.pair_indices = type_pair_idx[keep]

    log(logfile, f"\nEdge filtering: {n_removed}/{n_total} removed ({n_removed/max(n_total,1)*100:.1f}%)")

    # identify pair types with zero edges
    no_edge_pairs = set()
    all_pair_idx = torch.cat([g.pair_indices for g in dataset.graphs])
    for pi in range(num_type_pairs):
        if (all_pair_idx == pi).sum() == 0:
            no_edge_pairs.add(pi)

    # ---- CN neighbor check using subtype cutoffs ----
    # For cross-element checks (e.g. does Cd_core have O neighbors?), use the
    # minimum subtype cutoff involving that element pair as the bond threshold.
    cn_has_neighbor = {}
    for cn_type in sorted(set(t for t in atom_type_names if '_' in t)):
        be = base_element(cn_type)
        cn_idx = np.where(types_arr == cn_type)[0]
        for target in sorted(set(syms_arr)):
            if target == be:
                continue
            bp = canonical_pair(be, target)
            relevant_cuts = [v * BOHR_TO_ANGSTROM for k, v in subtype_cutoffs_bohr.items()
                             if cpair_to_base.get(k, k) == bp]
            cut_ang = min(relevant_cuts) if relevant_cuts else 4.0
            tidx = np.where(syms_arr == target)[0]
            cn_has_neighbor[(cn_type, target)] = any(
                np.any(dmat_ang[ai, tidx] < cut_ang) for ai in cn_idx)

    # ---- PMF-based initialization ----
    # PMF = -kT*ln(P(r)/r²) per subtype pair, fit Morse to the well.
    # Uses full distance distributions (not edge-filtered) for correct PMF shape.
    # CN correction: PMF overestimates alpha by ~sqrt(CN)
    training_temp = config.get('training_temperature', 1000)
    kT_eV = training_temp * 8.617333e-5

    stride = max(1, len(dataset.frame_data) // 200)
    pair_dists_ang = {i: [] for i in range(num_type_pairs)}
    for fi in range(0, len(dataset.frame_data), stride):
        pos_bohr = np.asarray(dataset.frame_data[fi][1])
        pos_a = pos_bohr * BOHR_TO_ANGSTROM
        dm = cdist(pos_a, pos_a)
        np.fill_diagonal(dm, np.inf)
        for i_t, t1 in enumerate(unique_types):
            idx1 = np.where(types_arr == t1)[0]
            for j_t in range(i_t, len(unique_types)):
                t2 = unique_types[j_t]
                idx2 = np.where(types_arr == t2)[0]
                cp = canonical_pair(t1, t2)
                pi = cpair_names.index(cp)
                if t1 == t2:
                    d = dm[np.ix_(idx1, idx2)]
                    d = d[np.triu_indices(len(idx1), k=1)]
                else:
                    d = dm[np.ix_(idx1, idx2)].ravel()
                pair_dists_ang[pi].extend(d[d < 8.0].tolist())

    init_D_e = np.full(num_type_pairs, 0.01 * EV_TO_HARTREE)
    init_alpha = np.full(num_type_pairs, 1.0 * BOHR_TO_ANGSTROM)
    init_r0 = np.full(num_type_pairs, 5.0 * ANGSTROM_TO_BOHR)

    # average cross-species CN per subtype (for PMF alpha correction)
    subtype_cn = {}
    for t in unique_types:
        be = base_element(t)
        if be in ORGANIC_ELEMENTS:
            continue
        t_idx = np.where(types_arr == t)[0]
        other_inorg = [e for e in sorted(inorganic) if e != be]
        cn_sum = 0
        for oe in other_inorg:
            bp = canonical_pair(be, oe)
            relevant_cuts = [v * BOHR_TO_ANGSTROM for k, v in subtype_cutoffs_bohr.items()
                             if cpair_to_base.get(k, k) == bp]
            cut_ang = min(relevant_cuts) if relevant_cuts else 3.5
            oe_idx = np.where(np.array([base_element(t) for t in types_arr]) == oe)[0]
            for ai in t_idx:
                cn_sum += np.sum(dmat_ang[ai, oe_idx] < cut_ang)
        subtype_cn[t] = cn_sum / max(len(t_idx), 1)
    for t, cn in sorted(subtype_cn.items()):
        log(logfile, f"  CN({t}): {cn:.1f}")

    log(logfile, f"\nPMF init (T={training_temp}K, kT={kT_eV:.4f}eV):")
    for i, cp in enumerate(cpair_names):
        bp = cpair_to_base[cp]
        b1, b2 = bp.split('-')
        if b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS:
            init_D_e[i] = 1e-6 * EV_TO_HARTREE
            continue
        if i in no_edge_pairs:
            continue
        t1, t2 = cp.split('-')
        no_bond = False
        if (t1, base_element(t2)) in cn_has_neighbor and not cn_has_neighbor[(t1, base_element(t2))]:
            no_bond = True
        if (t2, base_element(t1)) in cn_has_neighbor and not cn_has_neighbor[(t2, base_element(t1))]:
            no_bond = True
        if no_bond:
            continue

        dists_ang = np.array(pair_dists_ang[i])
        if len(dists_ang) < 100:
            continue

        # PMF: histogram -> P(r) -> correct for r² -> -kT*ln
        dr = 0.02
        bins = np.arange(1.5, 8.0 + dr, dr)
        r = 0.5 * (bins[:-1] + bins[1:])
        hist, _ = np.histogram(dists_ang, bins=bins)

        P = hist.astype(float) / (len(dists_ang) * dr)
        P_corr = P / (4 * np.pi * r**2)
        P_corr[P_corr <= 0] = 1e-30
        pmf = -kT_eV * np.log(P_corr)

        # fit Morse to PMF well
        mask = hist > 0
        if mask.sum() < 10:
            continue
        r_fit = r[mask]
        pmf_fit = pmf[mask]
        pmf_rel = pmf_fit - pmf_fit.min()
        r0_approx = r_fit[np.argmin(pmf_rel)]
        left = r_fit < r0_approx
        De_approx = min(pmf_rel[left].max(), 5.0) if left.sum() > 2 else 0.5
        De_approx = max(De_approx, 0.01)

        try:
            popt, _ = curve_fit(
                lambda r, De, a, r0: De * (1 - np.exp(-a * (r - r0)))**2,
                r_fit, pmf_rel,
                p0=[De_approx, 1.5, r0_approx],
                bounds=([0.001, 0.05, 1.5], [5.0, 5.0, 8.0]),
                maxfev=10000)
            D_e_pmf, alpha_pmf, r0_ang = popt

            # CN correction: PMF overestimates alpha by ~sqrt(CN)
            cn1 = subtype_cn.get(t1, 1.0)
            cn2 = subtype_cn.get(t2, 1.0)
            cn_avg = np.sqrt(max(cn1, 1.0) * max(cn2, 1.0))
            alpha_corr = alpha_pmf / np.sqrt(cn_avg)
            k_pmf = 2 * D_e_pmf * alpha_pmf**2
            D_e_corr = k_pmf / (2 * alpha_corr**2)

            init_r0[i] = r0_ang * ANGSTROM_TO_BOHR
            init_alpha[i] = alpha_corr * BOHR_TO_ANGSTROM
            init_D_e[i] = D_e_corr * EV_TO_HARTREE
            log(logfile, f"  {cp:>25}: r0={r0_ang:.3f} a={alpha_pmf:.3f}->{alpha_corr:.3f} "
                f"De={D_e_pmf:.4f}->{D_e_corr:.4f} (CN={cn_avg:.1f})")
        except Exception:
            mode_bin = np.argmax(hist)
            r0_ang = 0.5 * (bins[mode_bin] + bins[mode_bin + 1])
            init_r0[i] = r0_ang * ANGSTROM_TO_BOHR
            log(logfile, f"  {cp:>25}: r0={r0_ang:.3f} (PMF fit failed, r0 only)")

    #create model
    model = MorseModel(cpair_names, cpair_lookup, init_D_e, init_alpha, init_r0).to(device)

    #training setup
    num_elements = len(dataset.elements)
    n_total = len(dataset.graphs)
    n_val = max(1, int(n_total * config['validation_split']))
    n_train = n_total - n_val
    batch_size = config['batch_size']
    train_graphs = dataset.graphs[:n_train]
    val_loader = DataLoader(dataset.graphs[n_train:], batch_size=batch_size, shuffle=False)

    core_mask = None
    core_elements = config.get('core_elements', [])
    if core_elements:
        core_mask = torch.zeros(num_elements, dtype=torch.float64, device=device)
        for elem in core_elements:
            if elem in dataset.element_to_index:
                core_mask[dataset.element_to_index[elem]] = 1.0

    lr = config['learning_rate']
    max_epochs = config.get('max_epochs', 300)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)
    patience = config['convergence_patience']
    threshold = config['convergence_threshold']

    log(logfile, f"\nTrain: {n_train}, Val: {n_val}")

    #training
    best_val = float('inf')
    best_state = None
    no_improve = 0
    train_hist, val_hist = [], []

    log(logfile, f"\n{'Ep':>4} | {'Train':>10} | {'Val':>10} | {'LR':>10}")
    log(logfile, '-' * 45)

    for epoch in range(1, max_epochs + 1):
        if no_improve >= patience:
            log(logfile, f"\nConverged ({patience} epochs)")
            break

        model.train()
        sf, nb = 0.0, 0
        chunk = min(batch_size, n_train)
        chunks = list(range(0, max(1, n_train - chunk + 1), chunk))
        random.shuffle(chunks)

        for cs in chunks:
            ce = min(cs + chunk, n_train)
            batch = Batch.from_data_list(train_graphs[cs:ce]).to(device)
            optimizer.zero_grad(set_to_none=True)

            fp = model(batch)
            loss = force_rmse(fp, batch.dft_forces, batch.element_indices, core_mask)
            loss.backward()
            optimizer.step()
            sf += loss.item(); nb += 1

        tr = (sf / nb) * FORCE_AU_TO_EV_ANG

        model.eval()
        vt, vnb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                vt += force_rmse(model(batch), batch.dft_forces,
                                 batch.element_indices, core_mask).item()
                vnb += 1
        vr = (vt / vnb) * FORCE_AU_TO_EV_ANG
        train_hist.append(tr); val_hist.append(vr)

        if vr < best_val - threshold:
            best_val = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch <= 5 or epoch % 10 == 1 or no_improve == 0:
            log(logfile, f"{epoch:4d} | {tr:10.6f} | {vr:10.6f} | {optimizer.param_groups[0]['lr']:10.2e}")
        scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    log(logfile, f"\nBest val RMSE: {best_val:.6f} eV/A")
    print(f"Best val RMSE: {best_val:.6f} eV/A")

    #final params
    params = model.get_type_pair_params()
    log(logfile, f"\n{'='*60}\nFINAL PARAMETERS\n{'='*60}")
    for bp in sorted(set(cpair_to_base.values())):
        b1, b2 = bp.split('-')
        if b1 in ORGANIC_ELEMENTS and b2 in ORGANIC_ELEMENTS:
            continue
        for name in sorted(cpair_names):
            if cpair_to_base[name] != bp:
                continue
            p = params[name]
            if p['D_e'] < 1e-4:
                continue
            log(logfile, f"  {name:>25} D_e={p['D_e']:.6f} alpha={p['alpha']:.6f} r0={p['r0']:.6f}")

    #save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(), 'best_rmse': best_val,
        'config': config, 'type_names': unique_types,
        'atom_type_names': atom_type_names, 'cpair_names': cpair_names,
        'cpair_lookup': cpair_lookup, 'cpair_to_base': cpair_to_base,
        'train_rmse_history': train_hist, 'val_rmse_history': val_hist,
    }, os.path.join(results_dir, 'checkpoint.pt'))

    #plots
    from .plotting import generate_all_plots
    generate_all_plots(model, dataset, device, config, cpair_names=cpair_names,
                       cpair_to_base=cpair_to_base, train_rmse_history=train_hist,
                       val_rmse_history=val_hist, atom_type_names=atom_type_names,
                       output_dir=os.path.join(results_dir, 'plots'))

    #LAMMPS export
    from .lammps_export import export_lammps
    export_lammps(model=model, dataset=dataset, config=config,
                  atom_type_names=atom_type_names, unique_types=unique_types,
                  cpair_names=cpair_names, cpair_to_base=cpair_to_base,
                  output_dir=lammps_dir)

    elapsed = time.time() - start_time
    log(logfile, f"\n{'='*60}\nDone! {elapsed:.1f}s | Best: {best_val:.6f} eV/A\n{'='*60}")
    print(f"Done! {elapsed:.1f}s | Best: {best_val:.6f} eV/A")
    logfile.close()
