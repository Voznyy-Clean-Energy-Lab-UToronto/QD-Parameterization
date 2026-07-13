import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import DFTDataset, make_batch
from .models import sw_forces, bond_sigma
from .utils import BOHR_TO_ANGSTROM, EV_TO_HARTREE, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_triplet

BOLTZMANN_EV  = 8.617333e-5
GAMMA_INIT    = 1.2
P_INIT        = 4.0
Q_INIT        = 0.0


def load_config(filepath):
    with open(filepath) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(filepath))
    for dataset in config.get("datasets", []):
        if "xyz" in dataset and not os.path.isabs(dataset["xyz"]):
            dataset["xyz"] = os.path.normpath(os.path.join(config_dir, dataset["xyz"]))
    return config


def force_rmse(predicted, target):
    return torch.sqrt(((predicted - target) ** 2).mean())


def softplus_inv(y):

    return math.log(math.expm1(max(y, 1e-6)))


def _b_init(r0, sigma, cutoff):

    gap_sq = (r0 - cutoff) ** 2
    return (r0 / sigma) ** 4 / (1.0 + 4.0 * gap_sq / (sigma * r0))


def _a_init(r0, sigma, cutoff, b):

    bracket = b * (sigma / r0) ** 4 - 1.0
    return -1.0 / (bracket * math.exp(sigma / (r0 - cutoff)))


def gather_cos_samples(graphs, stride):
    samples = {}
    for graph in graphs[::stride]:
        positions = graph["positions"]
        for name, (centre, atom_a, atom_b) in graph["triplets"].items():
            vec_ca = positions[atom_a] - positions[centre]
            vec_cb = positions[atom_b] - positions[centre]
            cos_theta = (
                (vec_ca * vec_cb).sum(dim=1) / (vec_ca.norm(dim=1) * vec_cb.norm(dim=1))
            ).clamp(-1, 1)
            samples.setdefault(name, []).extend(cos_theta.tolist())
    return samples


def initial_cos_theta0(triplet_names, cos_samples):
    cos_theta0 = {}
    for name in triplet_names:
        samples = cos_samples.get(name, [])
        if len(samples) > 20:
            cos_theta0[name] = float(np.clip(np.mean(samples), -0.999, 0.999))
        else:
            cos_theta0[name] = -1.0 / 3.0
    return cos_theta0


def initial_lam(scales, cos_samples, temperature):

    lam_init = {}
    for bond in scales["cutoff"]:
        elem_a, elem_b = bond.split("-")
        cos_vals = []
        for name in [canonical_triplet(elem_a, elem_b, elem_b),
                     canonical_triplet(elem_b, elem_a, elem_a)]:
            cos_vals += cos_samples.get(name, [])

        if len(cos_vals) < 50:
            lam_init[bond] = 1e-4 * EV_TO_HARTREE
            continue

        var_cos    = float(np.var(cos_vals))
        sigma      = scales["sigma"][bond]
        cutoff     = scales["cutoff"][bond]
        r0         = scales["r0"][bond]
        decay_sq   = math.exp(GAMMA_INIT * sigma / (r0 - cutoff)) ** 2
        kT_hartree = BOLTZMANN_EV * temperature * EV_TO_HARTREE
        lam_init[bond] = kT_hartree / (2.0 * decay_sq * max(var_cos, 1e-8))

    return lam_init


def initialise_parameters(dataset, temperature, log):
    stride          = max(1, len(dataset.graphs) // 200)
    cos_samples     = gather_cos_samples(dataset.graphs, stride)
    cos_theta0_init = initial_cos_theta0(dataset.triplet_type_names, cos_samples)
    lam_init        = initial_lam(dataset.scales, cos_samples, temperature)

    bonds    = list(dataset.cutoffs_bohr.keys())
    triplets = list(dataset.triplet_type_names)


    eps = {
        bond: torch.tensor(
            dataset.scales["eps_init"][bond], dtype=torch.float64, requires_grad=True
        )
        for bond in bonds
    }


    raw_A, raw_B = {}, {}
    for bond in bonds:
        r0, sigma, cutoff = (dataset.scales[k][bond]
                             for k in ("r0", "sigma", "cutoff"))
        b0 = _b_init(r0, sigma, cutoff)
        a0 = _a_init(r0, sigma, cutoff, b0)
        raw_A[bond] = torch.tensor(math.log(max(a0, 1e-3)),
                                   dtype=torch.float64, requires_grad=True)
        raw_B[bond] = torch.tensor(softplus_inv(max(b0, 1e-3)),
                                   dtype=torch.float64, requires_grad=True)


    raw_p = {
        bond: torch.tensor(P_INIT, dtype=torch.float64, requires_grad=True)
        for bond in bonds
    }


    raw_q = {
        bond: torch.tensor(Q_INIT, dtype=torch.float64, requires_grad=True)
        for bond in bonds
    }


    raw_gamma = {
        bond: torch.tensor(softplus_inv(GAMMA_INIT), dtype=torch.float64, requires_grad=True)
        for bond in bonds
    }


    raw_lam = {
        bond: torch.tensor(
            math.log(max(lam_init[bond], 1e-10)), dtype=torch.float64, requires_grad=True
        )
        for bond in bonds
    }


    raw_theta0 = {
        triplet: torch.tensor(
            math.atanh(cos_theta0_init[triplet]), dtype=torch.float64, requires_grad=True
        )
        for triplet in triplets
    }

    log("\nInitial parameters:")
    log("  Trained:  eps, A, B, p, q, gamma, lambda, theta0")
    log("  Fixed:    cutoff (TAIL_FACTOR x RDF first-min);  sigma dynamic (pins well min at r0)")
    log(f"\n  {'Bond':>8}  {'eps (eV)':>10}  {'A':>8}  {'B':>8}  "
        f"{'p':>5}  {'q':>5}  {'gamma':>6}  {'lam (eV)':>10}  {'cutoff (Å)':>11}")
    for bond in sorted(bonds):
        cutoff_ang = dataset.scales["cutoff"][bond] * BOHR_TO_ANGSTROM
        log(f"  {bond:>8}  "
            f"{eps[bond].item()*HARTREE_TO_EV:>10.4f}  "
            f"{math.exp(raw_A[bond].item()):>8.4f}  "
            f"{F.softplus(raw_B[bond]).item():>8.4f}  "
            f"{raw_p[bond].item():>5.2f}  "
            f"{raw_q[bond].item():>5.2f}  "
            f"{F.softplus(raw_gamma[bond]).item():>6.4f}  "
            f"{math.exp(raw_lam[bond].item())*HARTREE_TO_EV:>10.4f}  "
            f"{cutoff_ang:>11.4f}")
    log("\n  cos_theta0 init (from DFT mean angle):")
    for name in sorted(cos_theta0_init):
        cos_val   = cos_theta0_init[name]
        theta_deg = math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))
        log(f"    {name}: cos_theta0={cos_val:.4f}  (theta0={theta_deg:.1f}°)")

    return {
        "eps":        eps,
        "raw_A":      raw_A,
        "raw_B":      raw_B,
        "raw_p":      raw_p,
        "raw_q":      raw_q,
        "raw_gamma":  raw_gamma,
        "raw_lam":    raw_lam,
        "raw_theta0": raw_theta0,
        "sigma":  dataset.scales["sigma"],
        "r0":     dataset.scales["r0"],
        "cutoff": dataset.scales["cutoff"],
    }


def epoch_rmse(batches_per_qd, params, train, optimizer=None):

    qd_order = list(range(len(batches_per_qd)))
    if train:
        random.shuffle(qd_order)
    qd_rmses = []
    for qd_idx in qd_order:
        qd_batches = batches_per_qd[qd_idx]
        batch_order = list(range(len(qd_batches)))
        if train:
            random.shuffle(batch_order)
        qd_total = 0.0
        for idx in batch_order:
            batch     = qd_batches[idx]
            predicted = sw_forces(batch, params)[batch["fit_mask"]]
            target    = batch["dft_forces"][batch["fit_mask"]]
            loss      = force_rmse(predicted, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for bond in params["eps"]:
                        params["eps"][bond].clamp_(min=1e-4)
                        params["raw_p"][bond].clamp_(min=0.0)
                        params["raw_q"][bond].clamp_(min=0.0)
            qd_total += loss.item()
        qd_rmses.append(qd_total / len(qd_batches))
    return sum(qd_rmses) / len(qd_rmses) * FORCE_AU_TO_EV_ANG


def train(config, output_dir):
    t_start = time.time()
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    logfile = open(os.path.join(results_dir, "training.log"), "w")

    def log(message):
        print(message)
        logfile.write(message + "\n")
        logfile.flush()

    training    = config["training"]
    temperature = training.get("training_temperature", 650)

    dataset = DFTDataset(config["datasets"], config["scope"],
                         first_n_frames=training.get("first_n_frames"))
    t_after_load = time.time()
    dataset.build_graphs(temperature=temperature)
    t_after_graphs = time.time()
    params = initialise_parameters(dataset, temperature, log)
    t_after_init = time.time()

    batch_size = training["batch_size"]
    val_frac   = training["validation_split"]
    train_batches_per_qd, val_batches_per_qd = [], []
    log("\nPer-QD train/val split (last {:.0%} of each QD → val):".format(val_frac))
    for ds_cfg, qd_graphs in zip(dataset._datasets, dataset.graphs_per_dataset):
        n_val_qd  = max(1, int(len(qd_graphs) * val_frac))
        qd_train  = qd_graphs[:-n_val_qd]
        qd_val    = qd_graphs[-n_val_qd:]
        train_batches_per_qd.append([make_batch(qd_train[i:i+batch_size])
                                     for i in range(0, len(qd_train), batch_size)])
        val_batches_per_qd.append([make_batch(qd_val[i:i+batch_size])
                                   for i in range(0, len(qd_val), batch_size)])
        log(f"  {ds_cfg['name']}: {len(qd_train)} train / {len(qd_val)} val frames")
    t_after_batches = time.time()

    optimizer = torch.optim.Adam(
        list(params["eps"].values())
        + list(params["raw_A"].values())
        + list(params["raw_B"].values())
        + list(params["raw_p"].values())
        + list(params["raw_q"].values())
        + list(params["raw_gamma"].values())
        + list(params["raw_lam"].values())
        + list(params["raw_theta0"].values()),
        lr=training["learning_rate"],
    )
    max_epochs = training["max_epochs"]
    min_lr     = float(training.get("min_lr", 1e-4))
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=min_lr
    )

    best_val    = float("inf")
    best_state  = None
    train_history, val_history = [], []

    log(f"\n{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>9} | {'Elapsed':>9} | imp")
    for epoch in range(1, max_epochs + 1):
        train_rmse = epoch_rmse(train_batches_per_qd, params, train=True, optimizer=optimizer)
        with torch.no_grad():
            val_rmse = epoch_rmse(val_batches_per_qd, params, train=False)
        scheduler.step()
        train_history.append(train_rmse)
        val_history.append(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t_start
        if val_rmse < best_val:
            best_val  = val_rmse
            best_state = {
                "eps":       {b: t.item() for b, t in params["eps"].items()},
                "raw_A":     {b: t.item() for b, t in params["raw_A"].items()},
                "raw_B":     {b: t.item() for b, t in params["raw_B"].items()},
                "raw_p":     {b: t.item() for b, t in params["raw_p"].items()},
                "raw_q":     {b: t.item() for b, t in params["raw_q"].items()},
                "raw_gamma": {b: t.item() for b, t in params["raw_gamma"].items()},
                "raw_lam":   {b: t.item() for b, t in params["raw_lam"].items()},
                "raw_theta0":{t: v.item() for t, v in params["raw_theta0"].items()},
            }
            log(f"{epoch:6d} | {train_rmse:10.6f} | {val_rmse:10.6f} | "
                f"{current_lr:9.2e} | {elapsed:8.1f}s | *")
        else:
            log(f"{epoch:6d} | {train_rmse:10.6f} | {val_rmse:10.6f} | "
                f"{current_lr:9.2e} | {elapsed:8.1f}s |  ")

    t_after_train = time.time()

    with torch.no_grad():
        for b, v in best_state["eps"].items():       params["eps"][b].fill_(v)
        for b, v in best_state["raw_A"].items():     params["raw_A"][b].fill_(v)
        for b, v in best_state["raw_B"].items():     params["raw_B"][b].fill_(v)
        for b, v in best_state["raw_p"].items():     params["raw_p"][b].fill_(v)
        for b, v in best_state["raw_q"].items():     params["raw_q"][b].fill_(v)
        for b, v in best_state["raw_gamma"].items(): params["raw_gamma"][b].fill_(v)
        for b, v in best_state["raw_lam"].items():   params["raw_lam"][b].fill_(v)
        for t, v in best_state["raw_theta0"].items():params["raw_theta0"][t].fill_(v)
    log(f"\nBest validation force RMSE: {best_val:.6f} eV/Angstrom")

    write_outputs(results_dir, dataset, params, best_val,
                  train_history, val_history, config, log)
    t_after_write = time.time()

    total = t_after_write - t_start
    phases = [
        ("Data loading",      t_after_load    - t_start),
        ("Graph building",    t_after_graphs  - t_after_load),
        ("Initial guess",     t_after_init    - t_after_graphs),
        ("Batch building",    t_after_batches - t_after_init),
        ("Training loop",     t_after_train   - t_after_batches),
        ("Output & plotting", t_after_write   - t_after_train),
    ]
    log(f"\n{'Phase':<22} {'Time':>8}  {'%':>5}")
    log("-" * 40)
    for name, secs in phases:
        log(f"  {name:<20} {secs:7.1f}s  {100*secs/total:4.1f}%")
    log("-" * 40)
    log(f"  {'Total':<20} {total:7.1f}s  100.0%")
    logfile.close()


def write_outputs(results_dir, dataset, params, best_val,
                  train_history, val_history, config, log):
    final = {}
    for bond in params["raw_A"]:
        a_val = float(params["cutoff"][bond] / bond_sigma(bond, params))
        final[bond] = {
            "eps_eV": params["eps"][bond].item() * HARTREE_TO_EV,
            "A":      math.exp(params["raw_A"][bond].item()),
            "B":      F.softplus(params["raw_B"][bond]).item(),
            "p":      params["raw_p"][bond].item(),
            "q":      params["raw_q"][bond].item(),
            "gamma":  F.softplus(params["raw_gamma"][bond]).item(),
            "lam_eV": math.exp(params["raw_lam"][bond].item()) * HARTREE_TO_EV,
            "a":      a_val,
        }
    final_cos_theta0 = {
        t: math.tanh(params["raw_theta0"][t].item())
        for t in params["raw_theta0"]
    }

    log(f"\n{'=' * 60}\nFITTED PARAMETERS — v33\n{'=' * 60}")
    log(f"  {'Bond':>8}  {'eps (eV)':>10}  {'A':>8}  {'B':>8}  "
        f"{'p':>6}  {'q':>6}  {'gamma':>7}  {'lam (eV)':>10}  {'a':>7}")
    for bond in sorted(final):
        d = final[bond]
        log(f"  {bond:>8}  {d['eps_eV']:>10.4f}  {d['A']:>8.4f}  {d['B']:>8.4f}  "
            f"  {d['p']:>6.3f}  {d['q']:>6.3f}  {d['gamma']:>7.4f}  {d['lam_eV']:>10.4f}  "
            f"{d['a']:>7.4f}")
    log("\n  Fitted cos_theta0 per triplet:")
    for triplet in sorted(final_cos_theta0):
        cos_val   = final_cos_theta0[triplet]
        theta_deg = math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))
        log(f"    {triplet}: cos_theta0={cos_val:.4f}  (theta0={theta_deg:.1f}°)")

    torch.save(
        {
            "final":            final,
            "final_cos_theta0": final_cos_theta0,
            "scales":           dataset.scales,
            "best_rmse":        best_val,
            "config":           config,
            "train_history":    train_history,
            "val_history":      val_history,
        },
        os.path.join(results_dir, "checkpoint.pt"),
    )

    from .plotting import plot_force_parity, plot_training, plot_sw_potentials
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_training(train_history, val_history, os.path.join(plots_dir, "training_curves.png"))
    plot_force_parity(dataset, params, os.path.join(plots_dir, "force_parity.png"))
    plot_sw_potentials(params, dataset.scales, os.path.join(plots_dir, "sw_potentials.png"))

    from .lammps_export import export_lammps
    export_lammps(results_dir, dataset, params, dataset.chemical_formula)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SW fitter v33: trains A, B, p, q, gamma, eps, lam, theta0, a per bond"
    )
    parser.add_argument("config", help="config YAML")
    parser.add_argument("--output-dir", default=None)
    args   = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    train(config, output_dir)
