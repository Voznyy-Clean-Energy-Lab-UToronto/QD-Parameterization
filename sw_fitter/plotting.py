import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

from .data import make_batch
from .models import sw_forces
from .utils import FORCE_AU_TO_EV_ANG


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
