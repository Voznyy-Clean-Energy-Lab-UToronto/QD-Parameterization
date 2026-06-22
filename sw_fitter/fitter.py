"""SW fitter: force-match epsilon, with geometry/angular scales frozen from the data.

See the module docstring in models.py for the parameter taxonomy. The fit itself
optimises only epsilon (the well depth); sigma, theta0, lambda, A, B are measured
from the trajectory and held fixed (unless training.naive enables the ablation).
"""
import os
import time
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch

from .data import DFTDataset, canonical_triplet, solve_sw_shape
from .models import SWModel, build_pair_infrastructure
from .utils import (HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE,
                    FORCE_AU_TO_EV_ANG, inverse_softplus)


def load_config(filepath):
    with open(filepath) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(filepath))
    for ds in config.get('datasets', []):
        if 'xyz' in ds and not os.path.isabs(ds['xyz']):
            ds['xyz'] = os.path.normpath(os.path.join(config_dir, ds['xyz']))
    return config


def force_rmse(predicted, target):
    return torch.sqrt(((predicted - target) ** 2).mean() + 1e-30)


def log(f, msg, stdout=False):
    f.write(msg + '\n'); f.flush()
    if stdout: print(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SW fitter: force-match epsilon')
    parser.add_argument('config', help='Config YAML')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    train(config, output_dir)


def train(config, output_dir):
    total_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    training = config['training']
    knn_edges = config['knn_edges']
    logfile = open(os.path.join(results_dir, 'training.log'), 'w')
    log(logfile, f'Device: {device}', stdout=True)

    # Load trajectories and build the graphs (edges, triplets, frozen scales).
    dataset = DFTDataset(config['datasets'], knn_edges,
                         first_n_frames=training.get('first_n_frames'))
    dataset.build_graphs(shell=knn_edges.get('shell', 1))

    if device.type == 'cuda':
        log(logfile, f'Pre-loading to GPU...', stdout=True)
        dataset.graphs = [g.to(device) for g in dataset.graphs]

    elements = dataset.elements
    pair_names, pair_lookup = build_pair_infrastructure(elements)
    num_pairs = len(pair_names)
    triplet_type_names = dataset.triplet_type_names
    num_tri = len(triplet_type_names)
    cutoffs_bohr = {pn: c for pn, c in dataset.cutoffs_bohr.items() if np.isfinite(c)}

    log(logfile, f'Types: {len(elements)}, Pairs: {num_pairs}, Triplets: {num_tri}', stdout=True)

    # ---- Parameter init: every scale comes from the data, none from literature. ----
    # FROZEN geometry: sigma <- bond length; theta0 <- angle-distribution mean.
    # FROZEN angular energy L=lambda*eps <- DFT angle-distribution VARIANCE by
    #   equipartition (force-matching cannot constrain lambda; the fluctuation can).
    # FIT: eps (depth) only -- it lives in the force budget.
    log(logfile, f'\nInit from data...', stdout=True)
    KB_EV = 8.617333e-5                                   # Boltzmann constant, eV/K
    T = training.get('training_temperature', 650)         # calibration temperature, K

    init_sigma = np.full(num_pairs, 4.0)                 # Bohr; set for scoped pairs only
    init_A = np.full(num_pairs, 7.05)                    # 2-body shape, from the RDF (data.py)
    init_B = np.full(num_pairs, 0.602)
    for pn, sig in dataset.sigma_bohr.items():
        if pn in pair_names:
            init_sigma[pair_names.index(pn)] = sig
            init_A[pair_names.index(pn)] = dataset.shape_A[pn]
            init_B[pair_names.index(pn)] = dataset.shape_B[pn]
    init_eps = np.full(num_pairs, 0.3 * EV_TO_HARTREE)   # depth seed; force-matching refines

    # gather the angle distribution per triplet type (from the graphs = the DFT data)
    tri_theta = {t: [] for t in range(num_tri)}
    stride = max(1, len(dataset.graphs) // 200)
    for fi in range(0, len(dataset.graphs), stride):
        g = dataset.graphs[fi]
        if g.triplet_type_idx.numel() == 0:
            continue
        tidx = g.triplet_type_idx.cpu().numpy()
        th = np.degrees(np.arccos(np.clip(g.tri_cos_theta.cpu().numpy(), -1, 1)))
        for t in np.unique(tidx):
            tri_theta[t].extend(th[tidx == t].tolist())

    # theta0 (mean) and L (equipartition from variance) per type/pair
    init_cos_theta0 = np.full(num_tri, -1.0 / 3.0)
    for t in range(num_tri):
        if len(tri_theta[t]) > 20:
            init_cos_theta0[t] = np.clip(np.cos(np.radians(np.mean(tri_theta[t]))), -0.99, 0.99)

    # L = lambda*eps by equipartition; facexp depends on the shape (sigma), so L is
    # recomputed in pass 2 when the shape is refined. Store (k_theta, sin^2 theta0) per pair.
    def L_from_equipartition(sig, cut, r0, k_theta, sin2):
        facexp = math.exp(1.2 * sig / (r0 - cut)) ** 2       # SW 3-body decay at the bond
        return (k_theta / (2.0 * sin2 * facexp)) * EV_TO_HARTREE   # -> Hartree

    pair_kth = {}                                            # pn -> (k_theta eV/rad^2, sin^2 theta0)
    init_L = np.full(num_pairs, 1e-4 * EV_TO_HARTREE)
    for pi, pn in enumerate(pair_names):
        if pn not in dataset.cutoffs_bohr:
            continue
        a, b = pn.split('-')
        angs = []
        for tn in (canonical_triplet(a, b, b), canonical_triplet(b, a, a)):
            ti = dataset.triplet_type_to_index.get(tn)
            if ti is not None:
                angs.extend(tri_theta[ti])
        if len(angs) < 50:
            continue
        angs = np.array(angs)
        th0 = np.radians(angs.mean())
        k_theta = KB_EV * T / np.radians(angs).var()         # eV/rad^2 (equipartition)
        sin2 = float(np.sin(th0) ** 2)
        pair_kth[pn] = (k_theta, sin2)
        init_L[pi] = L_from_equipartition(init_sigma[pi], dataset.cutoffs_bohr[pn],
                                          dataset.r0_bohr[pn], k_theta, sin2)
        log(logfile, f'  {pn:>12}: theta0={np.degrees(th0):.1f} k_theta={k_theta:.2f} '
                     f'-> L={init_L[pi]*HARTREE_TO_EV:.2f} eV', stdout=True)

    # Build the model. naive=False -> only eps trained, scales frozen (production).
    # naive=True  -> sigma/theta0/L/A/B also trainable from the data init (ABLATION:
    #                shows force-matching alone wrecks the physical scales).
    naive = training.get('naive', False)
    model = SWModel(pair_names, pair_lookup, cutoffs_bohr,
                    triplet_type_names=triplet_type_names,
                    init_eps=init_eps, init_sigma=init_sigma, init_A=init_A, init_B=init_B,
                    init_cos_theta0=init_cos_theta0, init_L=init_L, naive=naive).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "NAIVE (all scan params trainable)" if naive else "frozen scan + eps fit"
    log(logfile, f'\nMode: {mode}. Trainable params: {n_params}', stdout=True)

    n_total = len(dataset.graphs)
    n_val = max(1, int(n_total * training['validation_split']))
    n_train = n_total - n_val
    batch_size = training['batch_size']

    # Pre-build batches
    train_batches, val_batches = [], []
    for cs in range(0, n_train, batch_size):
        b = Batch.from_data_list(dataset.graphs[cs:min(cs+batch_size, n_train)])
        if device.type != 'cuda': b = b.to(device)
        train_batches.append(b)
    for cs in range(0, len(dataset.graphs) - n_train, batch_size):
        b = Batch.from_data_list(dataset.graphs[n_train + cs:min(n_train + cs + batch_size, n_total)])
        if device.type != 'cuda': b = b.to(device)
        val_batches.append(b)

    lr = training['learning_rate']
    patience = training['convergence_patience']
    threshold = training['convergence_threshold']
    train_hist, val_hist = [], []

    def fit_eps(tag):
        """Force-match the trainable parameters (eps only when frozen) to convergence."""
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=patience * 10, eta_min=lr * 0.01)
        best_val, best_state, no_improve, ep = float('inf'), None, 0, 0
        log(logfile, f"\n[{tag}] {'Ep':>5} | {'Train':>10} | {'Val':>10}", stdout=True)
        while no_improve < patience:
            ep += 1
            model.train()
            train_loss_sum, n_train_batches = 0.0, 0
            order = list(range(len(train_batches))); random.shuffle(order)
            for bi in order:
                opt.zero_grad(set_to_none=True)
                b = train_batches[bi]
                loss = force_rmse(model(b)[b.fit_mask], b.dft_forces[b.fit_mask])
                loss.backward(); opt.step()
                train_loss_sum += loss.item(); n_train_batches += 1
            train_rmse = (train_loss_sum / n_train_batches) * FORCE_AU_TO_EV_ANG
            model.eval()
            val_loss_sum, n_val_batches = 0.0, 0
            with torch.no_grad():
                for b in val_batches:
                    val_loss_sum += force_rmse(model(b)[b.fit_mask], b.dft_forces[b.fit_mask]).item()
                    n_val_batches += 1
            val_rmse = (val_loss_sum / n_val_batches) * FORCE_AU_TO_EV_ANG
            train_hist.append(train_rmse); val_hist.append(val_rmse)
            if val_rmse < best_val - threshold:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if ep <= 3 or no_improve == 0:
                log(logfile, f"[{tag}] {ep:5d} | {train_rmse:10.6f} | {val_rmse:10.6f}", stdout=True)
            sched.step()
        if best_state:
            model.load_state_dict(best_state)
        model.to(device)
        return best_val

    best_val = fit_eps("pass1")

    # ---- pass 2 (OPTIONAL, off by default): refine the 2-body shape from the RDF curvature
    # using the force-matched depth. This is the experiment that showed the SW form cannot
    # match the ionic Cd-Se well AND hold the structure (eps -> too shallow). Enable with
    # training.curvature_match: true. Production uses the Zhou II-VI shape (no pass 2).
    if not naive and training.get('curvature_match', False):
        eps_fit = F.softplus(model.raw_eps).detach().cpu().numpy()    # Hartree per pair
        kT_h = T * KB_EV * EV_TO_HARTREE
        log(logfile, "\nPass 2: curvature match using the force-matched depth:", stdout=True)
        with torch.no_grad():
            for pi, pn in enumerate(pair_names):
                if pn not in dataset.cutoffs_bohr or pn not in dataset.var_r_bohr:
                    continue
                r0, r_cut = dataset.r0_bohr[pn], dataset.cutoffs_bohr[pn]
                k_r = kT_h / dataset.var_r_bohr[pn]              # Hartree/Bohr^2 (RDF peak width)
                S_true = k_r * r0 ** 2 / max(eps_fit[pi], 1e-9)  # uses the TRUE depth, not PMF
                sh = solve_sw_shape(r0, r_cut, S_true)
                model.raw_sigma.data[pi] = inverse_softplus(sh['sigma'])
                model.raw_A.data[pi] = inverse_softplus(sh['A'])
                model.raw_B.data[pi] = inverse_softplus(sh['B'])
                if pn in pair_kth:
                    kth, sin2 = pair_kth[pn]
                    model.raw_L.data[pi] = inverse_softplus(
                        L_from_equipartition(sh['sigma'], r_cut, r0, kth, sin2))
                log(logfile, f"  {pn:>12}: eps={eps_fit[pi]*HARTREE_TO_EV:.3f}eV S_true={S_true:.0f} "
                             f"-> sigma={sh['sigma']*BOHR_TO_ANGSTROM:.3f} a={sh['a']:.3f} B={sh['B']:.3f}",
                    stdout=True)
        best_val = fit_eps("pass2")

    log(logfile, f'\nBest val RMSE: {best_val:.6f} eV/A', stdout=True)

    # Write outputs: checkpoint, plots, LAMMPS .sw file.
    try:
        pair_params, triplet_params = model.params_dict()

        log(logfile, f"\n{'='*60}\nFITTED 2-BODY + per-pair LAMBDA (data-derived theta0, Zhou mixing)\n{'='*60}")
        for pn in sorted(pair_params):
            p = pair_params[pn]
            if p['cutoff'] < 0.01: continue        # only scoped (active) pairs
            r0 = p['sigma'] * 1.28094              # A_FACTOR: r0 = sigma * 1.28094 (Zhou II-VI)
            log(logfile, f"  {pn:>12} eps={p['eps']:.4f} sig={p['sigma']:.3f} (r0={r0:.3f}) "
                         f"lambda={p['lambda']:.3f} cut={p['cutoff']:.2f}")

        torch.save({
            'model_state_dict': model.state_dict(), 'best_rmse': best_val,
            'config': config, 'pair_names': pair_names, 'pair_lookup': pair_lookup,
            'triplet_type_names': triplet_type_names,
            'train_rmse_history': train_hist, 'val_rmse_history': val_hist,
        }, os.path.join(results_dir, 'checkpoint.pt'))

        from .plotting import generate_all_plots
        generate_all_plots(model, dataset, device,
                           train_hist=train_hist, val_hist=val_hist,
                           output_dir=os.path.join(results_dir, 'plots'))

        from .lammps_export import export_lammps
        export_lammps(results_dir, pair_params, triplet_params, elements,
                      config.get('original_data'))
    except Exception as e:
        log(logfile, f'\nOutput phase FAILED: {e}', stdout=True)
        import traceback
        log(logfile, traceback.format_exc())

    total = time.time() - total_start
    log(logfile, f"\nDone! {total:.0f}s | Best: {best_val:.6f} eV/A", stdout=True)
    logfile.close()
