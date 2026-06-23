import math
import os
import random
import time

import numpy as np
import torch
import yaml

from .data import DFTDataset, make_batch
from .models import GAMMA, sw_forces
from .utils import EV_TO_HARTREE, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_triplet

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


#  Parameter initialisation
def gather_angle_samples(graphs, stride):
    samples = {}
    for graph in graphs[::stride]:
        positions = graph["positions"]
        for name, (centre, atom_a, atom_b) in graph["triplets"].items():
            vec_ca = positions[atom_a] - positions[centre]
            vec_cb = positions[atom_b] - positions[centre]
            cos_theta = (
                (vec_ca * vec_cb).sum(dim=1) / (vec_ca.norm(dim=1) * vec_cb.norm(dim=1))
            ).clamp(-1, 1)
            angles = torch.rad2deg(torch.arccos(cos_theta))
            samples.setdefault(name, []).extend(angles.tolist())
    return samples


def initial_cos_theta0(triplet_names, angle_samples):
    cos_theta0 = {}
    for name in triplet_names:
        angles = angle_samples.get(name, [])
        if len(angles) > 20:
            cos_theta0[name] = float(
                np.clip(np.cos(np.radians(np.mean(angles))), -0.99, 0.99)
            )
        else:
            cos_theta0[name] = -1.0 / 3.0  # ideal tetrahedral fallback
    return cos_theta0


def initial_angular_strength(scales, angle_samples, temperature):
    strength = {}
    for bond in scales["cutoff"]:
        subtype_a, subtype_b = bond.split("-")
        # the two triplet types whose angle the bond a-b controls (a at centre, b at centre)
        leg_types = [
            canonical_triplet(subtype_a, subtype_b, subtype_b),
            canonical_triplet(subtype_b, subtype_a, subtype_a),
        ]
        angles = []
        for triplet_name in leg_types:
            angles += angle_samples.get(triplet_name, [])
        if len(angles) < 50:
            strength[bond] = 1e-4 * EV_TO_HARTREE  # no data -> negligible
            continue

        angles = np.radians(np.array(angles))
        theta0 = angles.mean()
        k_theta = BOLTZMANN_EV * temperature / angles.var()  # eV per rad^2
        sin_sq = np.sin(theta0) ** 2
        # undo the SW 3-body exponential decay at the bond length, so L reproduces k_theta
        sigma, cutoff, r0 = (
            scales["sigma"][bond],
            scales["cutoff"][bond],
            scales["r0"][bond],
        )
        decay_factor = math.exp(GAMMA * sigma / (r0 - cutoff)) ** 2
        strength[bond] = (k_theta / (2.0 * sin_sq * decay_factor)) * EV_TO_HARTREE
    return strength


def initialise_parameters(dataset, temperature, log):
    stride = max(1, len(dataset.graphs) // 200)
    angle_samples = gather_angle_samples(dataset.graphs, stride)

    cos_theta0 = initial_cos_theta0(dataset.triplet_type_names, angle_samples)
    angular_strength = initial_angular_strength(
        dataset.scales, angle_samples, temperature
    )

    # eps (well depth) is the ONLY trained parameter: a tensor per bond that needs gradients.
    eps = {
        bond: torch.tensor(0.3 * EV_TO_HARTREE, dtype=torch.float64, requires_grad=True)
        for bond in dataset.cutoffs_bohr
    }

    log("\nInitial parameters (eps will be fit; the rest are frozen):")
    for bond in sorted(dataset.cutoffs_bohr):
        log(
            f"  {bond:>16}: L={angular_strength[bond] * HARTREE_TO_EV:.3f} eV (frozen), "
            f"eps=0.300 eV (start)"
        )

    return {
        "eps": eps,  # TRAINED (tensors)
        "sigma": dataset.scales["sigma"],  # frozen
        "cutoff": dataset.scales["cutoff"],  # frozen
        "A": dataset.scales["A"],  # frozen
        "B": dataset.scales["B"],  # frozen
        "L": angular_strength,  # frozen
        "cos_theta0": cos_theta0,  # frozen
    }


#  Training
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
    dataset = DFTDataset(
        config["datasets"], first_n_frames=training.get("first_n_frames")
    )
    dataset.build_graphs()
    params = initialise_parameters(
        dataset, training.get("training_temperature", 650), log
    )

    # Split frames into train/validation, then pre-concatenate them into batches.
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
        list(params["eps"].values()), lr=training["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training["convergence_patience"] * 10,
        eta_min=training["learning_rate"] * 0.01,
    )

    patience = training["convergence_patience"]
    threshold = training["convergence_threshold"]
    best_val = float("inf")
    best_eps = None
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
            best_eps = {bond: tensor.item() for bond, tensor in params["eps"].items()}
            epochs_without_improvement = 0
            log(f"{epoch:6d} | {train_rmse:10.6f} | {val_rmse:10.6f}")
        else:
            epochs_without_improvement += 1

    # restore the best-validation depths
    with torch.no_grad():
        for bond, value in best_eps.items():
            params["eps"][bond].fill_(value)
    log(f"\nBest validation force RMSE: {best_val:.6f} eV/Angstrom")

    write_outputs(
        results_dir, dataset, params, best_val, train_history, val_history, config, log
    )
    log(f"\nDone in {time.time() - start_time:.0f}s")
    logfile.close()


def write_outputs(
    results_dir, dataset, params, best_val, train_history, val_history, config, log
):
    final_eps = {bond: params["eps"][bond].item() for bond in params["eps"]}

    log(f"\n{'=' * 60}\nFITTED PARAMETERS (eV)\n{'=' * 60}")
    for bond in sorted(final_eps):
        eps_ev = final_eps[bond] * HARTREE_TO_EV
        lam = params["L"][bond] / final_eps[bond] if final_eps[bond] > 1e-12 else 0.0
        log(
            f"  {bond:>16}: eps={eps_ev:.4f}  lambda={lam:.3f}  "
            f"sigma={params['sigma'][bond] * 0.52917721:.3f} A"
        )

    torch.save(
        {
            "final_eps": final_eps,
            "scales": dataset.scales,
            "L": params["L"],
            "cos_theta0": params["cos_theta0"],
            "best_rmse": best_val,
            "config": config,
            "train_history": train_history,
            "val_history": val_history,
        },
        os.path.join(results_dir, "checkpoint.pt"),
    )

    from .plotting import plot_force_parity, plot_training

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_training(
        train_history, val_history, os.path.join(plots_dir, "training_curves.png")
    )
    plot_force_parity(dataset, params, os.path.join(plots_dir, "force_parity.png"))

    from .lammps_export import export_lammps

    export_lammps(results_dir, dataset, params)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SW fitter: force-match the well depths"
    )
    parser.add_argument("config", help="config YAML")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    train(config, output_dir)
