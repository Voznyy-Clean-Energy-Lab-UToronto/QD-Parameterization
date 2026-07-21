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
from .utils import EV_TO_HARTREE, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_pair

BOLTZMANN_EV  = 8.617333e-5
GAMMA_INIT    = 1.2
P_INIT        = 4.0
Q_INIT        = 0.0
A_INIT        = 7.049556
B_INIT        = 0.602
TETRAHEDRAL_COS = -1.0 / 3.0
SIGMA_RATIO   = 1.122
LAM_DEFAULT   = 1e-4

TRAINABLE = ("eps", "raw_A", "raw_B", "raw_p", "raw_q", "raw_gamma", "raw_lam", "raw_theta0")


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


def trainable(value):
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


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


def leg_decay(scales, bond):
    r0     = scales["r0"][bond]
    sigma  = r0 / SIGMA_RATIO
    cutoff = scales["cutoff"][bond]
    return math.exp(GAMMA_INIT * sigma / (r0 - cutoff))


def initial_lam(scales, cos_samples, triplet_names, temperature, eps_init_ev, eps_init_default):
    kT_hartree = BOLTZMANN_EV * temperature * EV_TO_HARTREE

    lam_init = {}
    for triplet_name in triplet_names:
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        cos_vals = cos_samples.get(triplet_name, [])
        if len(cos_vals) < 50:
            lam_init[triplet_name] = LAM_DEFAULT
            continue

        var_cos         = float(np.var(cos_vals))
        decay_ca        = leg_decay(scales, bond_ca)
        decay_cb        = leg_decay(scales, bond_cb)
        strength_target = kT_hartree / (2.0 * decay_ca * decay_cb * max(var_cos, 1e-8))

        eps_ca = eps_init_ev.get(bond_ca, eps_init_default) * EV_TO_HARTREE
        eps_cb = eps_init_ev.get(bond_cb, eps_init_default) * EV_TO_HARTREE
        lam_init[triplet_name] = strength_target / math.sqrt(eps_ca * eps_cb)

    return lam_init


def initial_cos_theta0(triplet_names, cos_samples):
    cos_theta0 = {}
    for name in triplet_names:
        samples = cos_samples.get(name, [])
        if len(samples) > 20:
            cos_theta0[name] = float(np.clip(np.mean(samples), -0.999, 0.999))
        else:
            cos_theta0[name] = TETRAHEDRAL_COS
    return cos_theta0


def initialise_parameters(dataset, temperature, eps_init_ev, eps_init_default):
    train_graphs    = [g for g in dataset.graphs if g.get("is_train", True)]
    stride          = max(1, len(train_graphs) // 200)
    cos_samples     = gather_cos_samples(train_graphs, stride)

    bonds    = list(dataset.cutoffs_bohr.keys())
    triplets = list(dataset.triplet_type_names)

    lam_init        = initial_lam(dataset.scales, cos_samples, triplets, temperature,
                                  eps_init_ev, eps_init_default)
    cos_theta0_init = initial_cos_theta0(triplets, cos_samples)

    eps        = {b: trainable(eps_init_ev.get(b, eps_init_default) * EV_TO_HARTREE) for b in bonds}
    raw_A      = {b: trainable(math.log(A_INIT))                  for b in bonds}
    raw_B      = {b: trainable(softplus_inv(B_INIT))              for b in bonds}
    raw_p      = {b: trainable(P_INIT)                            for b in bonds}
    raw_q      = {b: trainable(Q_INIT)                            for b in bonds}
    raw_gamma  = {b: trainable(softplus_inv(GAMMA_INIT))          for b in bonds}
    raw_lam    = {t: trainable(math.log(max(lam_init[t], 1e-10))) for t in triplets}
    raw_theta0 = {t: trainable(math.atanh(cos_theta0_init[t]))    for t in triplets}

    return {
        "eps":        eps,
        "raw_A":      raw_A,
        "raw_B":      raw_B,
        "raw_p":      raw_p,
        "raw_q":      raw_q,
        "raw_gamma":  raw_gamma,
        "raw_lam":    raw_lam,
        "raw_theta0": raw_theta0,
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
                         first_n_frames=training.get("first_n_frames"),
                         n_train=training.get("n_train_frames"),
                         val_stride=training.get("val_stride", 1))
    t_after_load = time.time()
    dataset.build_graphs()
    t_after_graphs = time.time()
    params = initialise_parameters(dataset, temperature,
                                   config.get("eps_init", {}),
                                   config.get("eps_init_default", 0.01))
    t_after_init = time.time()

    batch_size = training["batch_size"]
    val_frac   = training["validation_split"]
    train_batches_per_qd, val_batches_per_qd = [], []
    log("\nPer-QD train/val split (train = first n_train_frames, val = remainder):")
    for ds_cfg, qd_graphs in zip(dataset._datasets, dataset.graphs_per_dataset):
        if training.get("n_train_frames") is not None:
            qd_train = [g for g in qd_graphs if g["is_train"]]
            qd_val   = [g for g in qd_graphs if not g["is_train"]]
        else:
            n_val_qd = max(1, int(len(qd_graphs) * val_frac))
            qd_train = qd_graphs[:-n_val_qd]
            qd_val   = qd_graphs[-n_val_qd:]
        train_batches_per_qd.append([make_batch(qd_train[i:i+batch_size])
                                     for i in range(0, len(qd_train), batch_size)])
        val_batches_per_qd.append([make_batch(qd_val[i:i+batch_size])
                                   for i in range(0, len(qd_val), batch_size)])
        log(f"  {ds_cfg['name']}: {len(qd_train)} train / {len(qd_val)} val frames")
    t_after_batches = time.time()

    optimizer = torch.optim.Adam(
        [tensor for key in TRAINABLE for tensor in params[key].values()],
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

    val_every = int(training.get("val_every", 1))
    last_val  = float("inf")

    log(f"\n{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>9} | {'Elapsed':>9} | imp")
    for epoch in range(1, max_epochs + 1):
        train_rmse = epoch_rmse(train_batches_per_qd, params, train=True, optimizer=optimizer)
        scheduler.step()

        do_val   = (epoch % val_every == 0) or (epoch == max_epochs)
        improved = False
        if do_val:
            with torch.no_grad():
                last_val = epoch_rmse(val_batches_per_qd, params, train=False)
            if last_val < best_val:
                best_val   = last_val
                best_state = {key: {name: t.item() for name, t in params[key].items()}
                              for key in TRAINABLE}
                improved = True

        train_history.append(train_rmse)
        val_history.append(last_val)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t_start
        mark = "*" if improved else ("." if not do_val else " ")
        log(f"{epoch:6d} | {train_rmse:10.6f} | {last_val:10.6f} | "
            f"{current_lr:9.2e} | {elapsed:8.1f}s | {mark}")

    t_after_train = time.time()

    with torch.no_grad():
        for key in TRAINABLE:
            for name, value in best_state[key].items():
                params[key][name].fill_(value)
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
            "a":      a_val,
        }
    final_cos_theta0 = {
        t: math.tanh(params["raw_theta0"][t].item())
        for t in params["raw_theta0"]
    }
    final_lam = {}
    for triplet in params["raw_lam"]:
        centre, legs = triplet.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)
        lam_value = math.exp(params["raw_lam"][triplet].item())
        eps_ca    = params["eps"][bond_ca].item()
        eps_cb    = params["eps"][bond_cb].item()
        final_lam[triplet] = {
            "lam":         lam_value,
            "strength_eV": lam_value * math.sqrt(eps_ca * eps_cb) * HARTREE_TO_EV,
        }

    log(f"\n{'=' * 60}\nFITTED PARAMETERS\n{'=' * 60}")
    log(f"  {'Bond':>8}  {'eps (eV)':>10}  {'A':>8}  {'B':>8}  "
        f"{'p':>6}  {'q':>6}  {'gamma':>7}  {'a':>7}")
    for bond in sorted(final):
        d = final[bond]
        log(f"  {bond:>8}  {d['eps_eV']:>10.4f}  {d['A']:>8.4f}  {d['B']:>8.4f}  "
            f"  {d['p']:>6.3f}  {d['q']:>6.3f}  {d['gamma']:>7.4f}  "
            f"{d['a']:>7.4f}")
    log("\n  Fitted 3-body angular term per triplet "
        "(lambda dimensionless; strength = lambda*sqrt(eps_ca*eps_cb)):")
    log(f"    {'Triplet':>12}  {'lambda':>10}  {'strength (eV)':>14}  {'cos_theta0':>11}  {'theta0':>7}")
    for triplet in sorted(final_cos_theta0):
        cos_val   = final_cos_theta0[triplet]
        theta_deg = math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))
        angular   = final_lam[triplet]
        log(f"    {triplet:>12}  {angular['lam']:>10.4f}  {angular['strength_eV']:>14.5f}  "
            f"{cos_val:>11.4f}  {theta_deg:>6.1f}°")

    torch.save(
        {
            "final":            final,
            "final_cos_theta0": final_cos_theta0,
            "final_lam":        final_lam,
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
    plot_sw_potentials(params, os.path.join(plots_dir, "sw_potentials.png"))

    from .lammps_export import export_lammps
    export_lammps(results_dir, dataset, params, dataset.chemical_formula)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SW fitter: trains eps, A, B, p, q, gamma per bond and lambda, theta0 per triplet (cutoff fixed from RDF)"
    )
    parser.add_argument("config", help="config YAML")
    parser.add_argument("--output-dir", default=None)
    args   = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    train(config, output_dir)
