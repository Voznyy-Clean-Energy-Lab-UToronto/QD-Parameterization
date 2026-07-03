import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm

from .data import make_batch
from .models import sw_forces
from .utils import BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG, HARTREE_TO_EV, canonical_pair


def plot_training(train_history, val_history, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_history) + 1), train_history, label="train")
    ax.plot(range(1, len(val_history) + 1), val_history, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("force RMSE (eV/Angstrom)")
    ax.set_title(f"best validation RMSE = {min(val_history):.4f} eV/Angstrom"
                 if val_history else "training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
               "#2ca02c", "#d62728", "#17becf", "#bcbd22"]


def plot_sw_potentials(params, scales, filepath):
    """
    Two-panel figure:
      Left:  V2(r) in eV — one curve per bond type.
      Right: V3(theta) at r=r0 — one curve per triplet type.
    """
    eps        = params["eps"]
    raw_lam    = params["raw_lam"]
    raw_theta0 = params["raw_theta0"]
    sigma      = params["sigma"]
    r0         = params["r0"]
    cutoff     = params["cutoff"]

    bond_names    = sorted(eps.keys())
    triplet_names = sorted(raw_theta0.keys())

    fig, (ax_v2, ax_v3) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stillinger–Weber 2-body potential and 3-body angular term", fontsize=13)

    # --- V2(r) ---
    r_ang_grid = np.linspace(1.5, 7.5, 5000)
    v2_min_all = 0.0

    for color, bond in zip(PLOT_COLORS, bond_names):
        cutoff_bohr = float(cutoff[bond])
        sigma_bohr  = float(sigma[bond])
        cutoff_ang  = cutoff_bohr * BOHR_TO_ANGSTROM
        eps_ev      = float(eps[bond].item()) * HARTREE_TO_EV
        A_val       = math.exp(params["raw_A"][bond].item())
        B_val       = F.softplus(params["raw_B"][bond]).item()
        p_val       = params["raw_p"][bond].item()
        q_val       = params["raw_q"][bond].item()

        r_plot     = r_ang_grid[r_ang_grid < cutoff_ang * 0.99]
        r_bohr     = r_plot / BOHR_TO_ANGSTROM
        sig_over_r = sigma_bohr / r_bohr
        bracket    = B_val * sig_over_r**p_val - sig_over_r**q_val
        gap        = r_bohr - cutoff_bohr
        decay      = np.exp(np.clip(sigma_bohr / gap, -500, 0))
        v2_vals    = eps_ev * A_val * bracket * decay
        v2_min_all = min(v2_min_all, v2_vals.min())

        ax_v2.plot(r_plot, v2_vals, color=color, lw=2.0,
                   label=f"{bond}  (p={p_val:.2f}, q={q_val:.2f})")
        r_eq_ang = r_plot[np.argmin(v2_vals)]
        ax_v2.plot(r_eq_ang, v2_vals.min(), "o", color=color, ms=5, zorder=5)

    ax_v2.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax_v2.set_xlabel(r"$r$  (Å)", fontsize=12)
    ax_v2.set_ylabel(r"$V_2(r)$  (eV)", fontsize=12)
    ax_v2.set_title("Two-body term V₂(r)", fontsize=11)
    ax_v2.set_xlim(1.5, 7.5)
    depth = abs(v2_min_all)
    ax_v2.set_ylim(v2_min_all * 1.15 if v2_min_all < 0 else -0.1, depth * 0.5)
    ax_v2.legend(fontsize=9)
    ax_v2.grid(alpha=0.3)

    # --- V3(theta) at r = r0 ---
    theta_deg  = np.linspace(30, 180, 1000)
    cos_theta  = np.cos(np.radians(theta_deg))
    v3_max_all = 0.0

    for color, triplet_name in zip(PLOT_COLORS, triplet_names):
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        if bond_ca not in raw_lam or bond_cb not in raw_lam:
            continue
        lam_ca   = math.exp(raw_lam[bond_ca].item())
        lam_cb   = math.exp(raw_lam[bond_cb].item())
        strength = math.sqrt(lam_ca * lam_cb) * HARTREE_TO_EV

        r0_ca    = float(r0[bond_ca])
        r0_cb    = float(r0[bond_cb])
        cut_ca   = float(cutoff[bond_ca])
        cut_cb   = float(cutoff[bond_cb])
        sig_ca   = float(sigma[bond_ca])
        sig_cb   = float(sigma[bond_cb])
        gamma_ca = F.softplus(params["raw_gamma"][bond_ca]).item()
        gamma_cb = F.softplus(params["raw_gamma"][bond_cb]).item()
        decay_ca = math.exp(gamma_ca * sig_ca / (r0_ca - cut_ca))
        decay_cb = math.exp(gamma_cb * sig_cb / (r0_cb - cut_cb))

        cos0       = math.tanh(float(raw_theta0[triplet_name].item()))
        theta0_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos0))))
        v3 = strength * decay_ca * decay_cb * (cos_theta - cos0) ** 2
        v3_max_all = max(v3_max_all, v3.max())

        readable_label = fr"{leg_a}–{centre}–{leg_b}  ($\theta_0={theta0_deg:.1f}°$)"
        ax_v3.plot(theta_deg, v3, color=color, lw=2.0, label=readable_label)
        ax_v3.axvline(theta0_deg, color=color, lw=0.8, ls=":", alpha=0.5)

    ax_v3.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax_v3.set_xlabel(r"Bond angle  $\theta$  (°)", fontsize=12)
    ax_v3.set_ylabel(r"$V_3(\theta)$  (eV)", fontsize=12)
    ax_v3.set_title(r"Three-body term at $r = r_\mathrm{eq}$", fontsize=12)
    ax_v3.set_xlim(30, 180)
    ax_v3.set_ylim(-0.05, max(v3_max_all * 1.15, 0.1))
    ax_v3.legend(fontsize=9)
    ax_v3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {filepath}")


def plot_force_parity(dataset, params, filepath):
    predicted_chunks, target_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(dataset.graphs), 16):
            batch = make_batch(dataset.graphs[start:start+16])
            mask  = batch["fit_mask"]
            predicted_chunks.append(sw_forces(batch, params)[mask].numpy())
            target_chunks.append(batch["dft_forces"][mask].numpy())

    predicted = np.concatenate(predicted_chunks).ravel() * FORCE_AU_TO_EV_ANG
    target    = np.concatenate(target_chunks).ravel()    * FORCE_AU_TO_EV_ANG
    rmse      = np.sqrt(np.mean((predicted - target) ** 2))
    r2        = 1.0 - np.sum((predicted - target) ** 2) / np.sum((target - target.mean()) ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(target, predicted, gridsize=80, mincnt=1, cmap="viridis", norm=LogNorm())
    limit = max(abs(predicted.max()), abs(predicted.min()),
                abs(target.max()),    abs(target.min())) * 1.05
    ax.plot([-limit, limit], [-limit, limit], "r--", lw=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT force (eV/Angstrom)")
    ax.set_ylabel("SW force (eV/Angstrom)")
    ax.set_title("force parity")
    ax.text(0.04, 0.96, f"RMSE = {rmse:.4f} eV/Å\n$R^2$ = {r2:.4f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
