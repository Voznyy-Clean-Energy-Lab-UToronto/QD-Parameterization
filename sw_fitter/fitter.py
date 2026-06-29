import math
import os
import random
import time

import numpy as np
import torch
import yaml

from .data import DFTDataset, make_batch
from .models import (
    A_from_rmin_float, B_from_rmin, GAMMA, rmin_from_raw, sw_forces
)
from .utils import BOHR_TO_ANGSTROM, EV_TO_HARTREE, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_triplet

BOLTZMANN_EV = 8.617333e-5  # eV per Kelvin


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
    """Mean cos(theta) from DFT data. Tetrahedral fallback if too few samples."""
    cos_theta0 = {}
    for name in triplet_names:
        samples = cos_samples.get(name, [])
        if len(samples) > 20:
            cos_theta0[name] = float(np.clip(np.mean(samples), -0.999, 0.999))
        else:
            cos_theta0[name] = -1.0 / 3.0  # tetrahedral
    return cos_theta0


def initial_lam(scales, cos_samples, temperature):
    lam_init = {}
    for bond in scales["cutoff"]:
        elem_a, elem_b = bond.split("-")
        leg_types = [
            canonical_triplet(elem_a, elem_b, elem_b),
            canonical_triplet(elem_b, elem_a, elem_a),
        ]
        cos_vals = []
        for triplet_name in leg_types:
            cos_vals += cos_samples.get(triplet_name, [])

        if len(cos_vals) < 50:
            lam_init[bond] = 1e-4 * EV_TO_HARTREE
            continue

        cos_vals = np.array(cos_vals)
        var_cos = float(np.var(cos_vals))
        sigma = scales["sigma"][bond]
        cutoff = scales["cutoff"][bond]
        r0 = scales["r0"][bond]
        decay_sq = math.exp(GAMMA * sigma / (r0 - cutoff)) ** 2
        kT_hartree = BOLTZMANN_EV * temperature * EV_TO_HARTREE
        lam_init[bond] = kT_hartree / (2.0 * decay_sq * max(var_cos, 1e-8))

    return lam_init


def initialise_parameters(dataset, temperature, log):
    stride = max(1, len(dataset.graphs) // 200)
    cos_samples = gather_cos_samples(dataset.graphs, stride)
    cos_theta0_init = initial_cos_theta0(dataset.triplet_type_names, cos_samples)
    lam_init = initial_lam(dataset.scales, cos_samples, temperature)

    # eps: trained. Init from bond-length variance equipartition (scales["eps_init"]).
    eps = {
        bond: torch.tensor(
            dataset.scales["eps_init"][bond], dtype=torch.float64, requires_grad=True
        )
        for bond in dataset.cutoffs_bohr
    }

    # raw_rmin: trained. r_min = sigma + (cutoff-sigma)*sigmoid(raw_rmin)
    # Init so r_min = r0 (RDF first-shell peak)
    # B(r_min) and A(r_min) are derived analytically
    raw_rmin = {}
    for bond in dataset.cutoffs_bohr:
        sigma_val = dataset.scales["sigma"][bond]
        cutoff_val = dataset.scales["cutoff"][bond]
        r0_val = dataset.scales["r0"][bond]
        frac = (r0_val - sigma_val) / (cutoff_val - sigma_val)
        frac = max(1e-4, min(1 - 1e-4, frac))
        raw_rmin[bond] = torch.tensor(
            math.log(frac / (1.0 - frac)),  # logit: sigmoid^{-1}(frac)
            dtype=torch.float64,
            requires_grad=True,
        )

    # raw_lam: trained. lam = exp(raw_lam) per bond (Hartree)
    # Init so lam = equipartition estimate
    raw_lam = {
        bond: torch.tensor(
            math.log(max(lam_init[bond], 1e-10)),
            dtype=torch.float64,
            requires_grad=True,
        )
        for bond in dataset.cutoffs_bohr
    }
    raw_theta0 = {
        triplet: torch.tensor(
            math.atanh(cos_theta0_init[triplet]),
            dtype=torch.float64,
            requires_grad=True,
        )
        for triplet in dataset.triplet_type_names
    }

    log("\nInitial parameters:")
    log("  Trained:  eps (equipartition), raw_rmin → r_min (RDF peak),")
    log("            raw_lam → lambda (equipartition init), raw_theta0 (DFT mean angle)")
    log("  Frozen:   sigma, cutoff (from RDF)")
    log("  Derived:  B = B(r_min) from equilibrium condition; A = A(r_min) from well-depth condition")
    log(f"\n  {'Bond':>8}  {'eps_init (eV)':>14}  {'r_min_init (A)':>15}  {'B_init':>8}  {'A_init':>8}  {'lam_init (eV)':>14}")
    for bond in sorted(dataset.cutoffs_bohr):
        eps_ev = eps[bond].item() * HARTREE_TO_EV
        sigma_val = dataset.scales["sigma"][bond]
        cutoff_val = dataset.scales["cutoff"][bond]
        r0_val = dataset.scales["r0"][bond]
        r_min_ang = r0_val * BOHR_TO_ANGSTROM
        B_init = float(B_from_rmin(r0_val, sigma_val, cutoff_val))
        A_init = A_from_rmin_float(r0_val, sigma_val, cutoff_val)
        lam_ev = lam_init[bond] * HARTREE_TO_EV
        log(f"  {bond:>8}  {eps_ev:>14.4f}  {r_min_ang:>15.4f}  {B_init:>8.4f}  {A_init:>8.4f}  {lam_ev:>14.4f}")

    log("\n  Initial cos_theta0 per triplet (from DFT mean):")
    for name in sorted(cos_theta0_init):
        cos_val = cos_theta0_init[name]
        theta_deg = math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))
        log(f"    {name}: cos_theta0={cos_val:.4f}  (theta0={theta_deg:.1f})")

    return {
        "eps": eps,               # trained tensors
        "raw_rmin": raw_rmin,     # trained tensors; r_min = sigma+(cutoff-sigma)*sigmoid(raw_rmin)
        "raw_lam": raw_lam,       # trained tensors; lam = exp(raw_lam) in Hartree
        "raw_theta0": raw_theta0, # trained tensors; cos_theta0 = tanh(raw_theta0)
        "sigma": dataset.scales["sigma"],   # frozen plain-float dict (Bohr)
        "r0": dataset.scales["r0"],         # frozen plain-float dict (Bohr)
        "cutoff": dataset.scales["cutoff"], # frozen plain-float dict (Bohr)
    }


def epoch_rmse(batches, params, train, optimizer=None):
    order = list(range(len(batches)))
    if train:
        random.shuffle(order)
    total = 0.0
    for i in order:
        batch = batches[i]
        predicted = sw_forces(batch, params)[batch["fit_mask"]]
        target = batch["dft_forces"][batch["fit_mask"]]
        loss = force_rmse(predicted, target)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for bond in params["eps"]:
                    params["eps"][bond].clamp_(min=1e-4)
        total += loss.item()
    return total / len(batches) * FORCE_AU_TO_EV_ANG


def train(config, output_dir):
    start_time = time.time()
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    logfile = open(os.path.join(results_dir, "training.log"), "w")

    def log(message):
        print(message)
        logfile.write(message + "\n")
        logfile.flush()

    training = config["training"]
    temperature = training.get("training_temperature", 650)

    dataset = DFTDataset(config["datasets"], first_n_frames=training.get("first_n_frames"))
    dataset.build_graphs(temperature=temperature)
    params = initialise_parameters(dataset, temperature, log)

    batch_size = training["batch_size"]
    n_val = max(1, int(len(dataset.graphs) * training["validation_split"]))
    train_graphs = dataset.graphs[:-n_val]
    val_graphs = dataset.graphs[-n_val:]
    atoms = dataset.atoms_per_frame
    train_batches = [
        make_batch(train_graphs[i : i + batch_size], atoms)
        for i in range(0, len(train_graphs), batch_size)
    ]
    val_batches = [
        make_batch(val_graphs[i : i + batch_size], atoms)
        for i in range(0, len(val_graphs), batch_size)
    ]
    log(
        f"\n{len(train_graphs)} train / {len(val_graphs)} val frames, "
        f"{len(train_batches)} train batches"
    )

    optimizer = torch.optim.Adam(
        list(params["eps"].values())
        + list(params["raw_rmin"].values())
        + list(params["raw_lam"].values())
        + list(params["raw_theta0"].values()),
        lr=training["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training["convergence_patience"] * 10,
        eta_min=training["learning_rate"] * 0.01,
    )

    patience = training["convergence_patience"]
    threshold = training["convergence_threshold"]
    best_val = float("inf")
    best_eps = best_raw_rmin = best_raw_lam = best_raw_theta0 = None
    epochs_without_improvement = 0
    train_history, val_history = [], []

    log(f"\n{'Epoch':>6} | {'Train':>10} | {'Val':>10}")
    epoch = 0
    while epochs_without_improvement < patience:
        epoch += 1
        train_rmse = epoch_rmse(train_batches, params, train=True, optimizer=optimizer)
        with torch.no_grad():
            val_rmse = epoch_rmse(val_batches, params, train=False)
        scheduler.step()
        train_history.append(train_rmse)
        val_history.append(val_rmse)

        if val_rmse < best_val - threshold:
            best_val = val_rmse
            best_eps = {b: t.item() for b, t in params["eps"].items()}
            best_raw_rmin = {b: t.item() for b, t in params["raw_rmin"].items()}
            best_raw_lam = {b: t.item() for b, t in params["raw_lam"].items()}
            best_raw_theta0 = {t: v.item() for t, v in params["raw_theta0"].items()}
            epochs_without_improvement = 0
            log(f"{epoch:6d} | {train_rmse:10.6f} | {val_rmse:10.6f}")
        else:
            epochs_without_improvement += 1

    with torch.no_grad():
        for b, v in best_eps.items():        params["eps"][b].fill_(v)
        for b, v in best_raw_rmin.items():   params["raw_rmin"][b].fill_(v)
        for b, v in best_raw_lam.items():    params["raw_lam"][b].fill_(v)
        for t, v in best_raw_theta0.items(): params["raw_theta0"][t].fill_(v)
    log(f"\nBest validation force RMSE: {best_val:.6f} eV/Angstrom")

    write_outputs(results_dir, dataset, params, best_val, train_history, val_history, config, log)
    log(f"\nDone in {time.time() - start_time:.0f}s")
    logfile.close()


def write_outputs(results_dir, dataset, params, best_val, train_history, val_history, config, log):
    final_eps = {b: params["eps"][b].item() for b in params["eps"]}
    final_raw_rmin = {b: params["raw_rmin"][b].item() for b in params["raw_rmin"]}
    final_raw_lam = {b: params["raw_lam"][b].item() for b in params["raw_lam"]}
    final_lam = {b: math.exp(v) for b, v in final_raw_lam.items()}
    final_raw_theta0 = {t: params["raw_theta0"][t].item() for t in params["raw_theta0"]}
    final_cos_theta0 = {t: math.tanh(v) for t, v in final_raw_theta0.items()}

    final_rmin, final_B, final_A = {}, {}, {}
    for bond in final_raw_rmin:
        sigma_val = params["sigma"][bond]
        cutoff_val = params["cutoff"][bond]
        r_min = sigma_val + (cutoff_val - sigma_val) / (1.0 + math.exp(-final_raw_rmin[bond]))
        final_rmin[bond] = r_min
        final_B[bond] = float(B_from_rmin(r_min, sigma_val, cutoff_val))
        final_A[bond] = A_from_rmin_float(r_min, sigma_val, cutoff_val)

    log(f"\n{'=' * 60}\nFITTED PARAMETERS — v27\n{'=' * 60}")
    for bond in sorted(final_eps):
        eps_ev = final_eps[bond] * HARTREE_TO_EV
        lam_ev = final_lam[bond] * HARTREE_TO_EV
        r_min_ang = final_rmin[bond] * BOHR_TO_ANGSTROM
        log(
            f"  {bond:>8}: eps={eps_ev:.4f} eV  r_min={r_min_ang:.4f} A  "
            f"B={final_B[bond]:.4f}  A={final_A[bond]:.4f}  lambda={lam_ev:.4f} eV"
        )
    log("\n  Fitted cos_theta0 per triplet:")
    for triplet in sorted(final_cos_theta0):
        cos_val = final_cos_theta0[triplet]
        theta_deg = math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0))))
        log(f"    {triplet}: cos_theta0={cos_val:.4f}  (theta0={theta_deg:.1f})")

    torch.save(
        {
            "final_eps": final_eps,
            "final_raw_rmin": final_raw_rmin,
            "final_rmin": final_rmin,
            "final_B": final_B,
            "final_A": final_A,
            "final_raw_lam": final_raw_lam,
            "final_lam": final_lam,
            "final_raw_theta0": final_raw_theta0,
            "final_cos_theta0": final_cos_theta0,
            "scales": dataset.scales,
            "best_rmse": best_val,
            "config": config,
            "train_history": train_history,
            "val_history": val_history,
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
    export_lammps(results_dir, dataset, params)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SW fitter v27: trains eps, r_min, lambda, cos_theta0; B and A derived analytically"
    )
    parser.add_argument("config", help="config YAML")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    train(config, output_dir)
