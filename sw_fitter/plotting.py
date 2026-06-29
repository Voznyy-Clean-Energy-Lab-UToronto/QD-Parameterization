import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

from .data import make_batch
from .models import A_from_rmin_float, B_from_rmin, GAMMA, P, sw_forces
from .utils import BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_pair


def plot_training(train_history, val_history, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_history) + 1), train_history, label="train")
    ax.plot(range(1, len(val_history) + 1), val_history, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("force RMSE (eV/Angstrom)")
    ax.set_title(f"best validation RMSE = {min(val_history):.4f} eV/Angstrom")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_sw_potentials(params, scales, filepath):
    """
    Two-panel figure:
      Left:  V2(r) in eV vs r in Angstrom, one curve per bond type.
      Right: V3(theta) at r=r0 for both legs, one curve per triplet type.
    """
    eps = params["eps"]
    raw_rmin = params["raw_rmin"]
    raw_lam = params["raw_lam"]
    raw_theta0 = params["raw_theta0"]
    sigma = params["sigma"]
    r0 = params["r0"]
    cutoff = params["cutoff"]

    bond_names = sorted(eps.keys())
    triplet_names = sorted(raw_theta0.keys())

    fig, (ax_v2, ax_v3) = plt.subplots(1, 2, figsize=(14, 6))

    # --- V2(r) ---
    # Shared x-grid from the shortest first-shell start to the longest cutoff.
    r_min_ang = min(0.82 * float(r0[b]) * BOHR_TO_ANGSTROM for b in bond_names)
    r_max_ang = max(float(cutoff[b]) * BOHR_TO_ANGSTROM for b in bond_names)
    r_ang_grid = np.linspace(r_min_ang, r_max_ang, 800)

    v2_min_all = 0.0
    for bond in bond_names:
        cutoff_bohr = float(cutoff[bond])
        sigma_bohr = float(sigma[bond])
        cutoff_ang = cutoff_bohr * BOHR_TO_ANGSTROM
        eps_ev = float(eps[bond].item()) * HARTREE_TO_EV
        raw = raw_rmin[bond].item()
        r_min_bohr = sigma_bohr + (cutoff_bohr - sigma_bohr) / (1.0 + math.exp(-raw))
        B_val = float(B_from_rmin(r_min_bohr, sigma_bohr, cutoff_bohr))
        A_val = A_from_rmin_float(r_min_bohr, sigma_bohr, cutoff_bohr)

        v2 = np.zeros_like(r_ang_grid)
        inside = r_ang_grid < cutoff_ang
        r_bohr_inside = r_ang_grid[inside] / BOHR_TO_ANGSTROM
        sigma_over_r = sigma_bohr / r_bohr_inside
        bracket = B_val * sigma_over_r ** P - 1.0
        gap = r_bohr_inside - cutoff_bohr
        decay = np.exp(np.clip(sigma_bohr / gap, -500, 0))
        v2[inside] = eps_ev * A_val * bracket * decay
        # V2 = 0 beyond cutoff — already initialised to zero above.

        v2_min_all = min(v2_min_all, v2.min())
        ax_v2.plot(r_ang_grid, v2, label=bond, lw=1.8)

    ax_v2.axhline(0, color="k", lw=0.7, ls="--")
    ax_v2.set_xlabel("r (Å)")
    ax_v2.set_ylabel("V₂(r) (eV)")
    ax_v2.set_title("Two-body potential V₂(r)")
    ax_v2.set_ylim(v2_min_all * 1.15, abs(v2_min_all) * 0.5)
    ax_v2.legend(fontsize=8)
    ax_v2.grid(alpha=0.3)

    # --- V3(theta) at r = r0 for both legs ---
    theta_deg = np.linspace(0.5, 179.5, 360)
    cos_theta = np.cos(np.radians(theta_deg))
    for triplet_name in triplet_names:
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        if bond_ca not in raw_lam or bond_cb not in raw_lam:
            continue
        lam_ca = math.exp(raw_lam[bond_ca].item())
        lam_cb = math.exp(raw_lam[bond_cb].item())
        strength = math.sqrt(lam_ca * lam_cb) * HARTREE_TO_EV

        r0_ca = float(r0[bond_ca])
        r0_cb = float(r0[bond_cb])
        cutoff_ca = float(cutoff[bond_ca])
        cutoff_cb = float(cutoff[bond_cb])
        sigma_ca = float(sigma[bond_ca])
        sigma_cb = float(sigma[bond_cb])
        decay_ca = math.exp(GAMMA * sigma_ca / (r0_ca - cutoff_ca))
        decay_cb = math.exp(GAMMA * sigma_cb / (r0_cb - cutoff_cb))

        cos0 = math.tanh(float(raw_theta0[triplet_name].item()))
        v3 = strength * decay_ca * decay_cb * (cos_theta - cos0) ** 2

        ax_v3.plot(theta_deg, v3, label=triplet_name, lw=1.2)

    ax_v3.set_xlabel("θ (degrees)")
    ax_v3.set_ylabel("V₃(θ) at r=r₀ (eV/triplet)")
    ax_v3.set_title("Three-body potential V₃(θ) at r = r₀")
    ax_v3.legend(fontsize=6, ncol=2)
    ax_v3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  wrote {filepath}")


def plot_force_parity(dataset, params, filepath):
    predicted_chunks, target_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(dataset.graphs), 16):
            batch = make_batch(
                dataset.graphs[start : start + 16], dataset.atoms_per_frame
            )
            mask = batch["fit_mask"]
            predicted_chunks.append(sw_forces(batch, params)[mask].numpy())
            target_chunks.append(batch["dft_forces"][mask].numpy())

    predicted = np.concatenate(predicted_chunks).ravel() * FORCE_AU_TO_EV_ANG
    target = np.concatenate(target_chunks).ravel() * FORCE_AU_TO_EV_ANG
    rmse = np.sqrt(np.mean((predicted - target) ** 2))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(target, predicted, gridsize=80, mincnt=1, cmap="viridis", norm=LogNorm())
    limit = np.percentile(np.abs(np.concatenate([target, predicted])), 99.5) * 1.1
    ax.plot([-limit, limit], [-limit, limit], "r--", lw=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT force (eV/Angstrom)")
    ax.set_ylabel("SW force (eV/Angstrom)")
    ax.set_title(f"force parity, RMSE = {rmse:.4f} eV/Angstrom")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
