"""Plotting: training curves, force parity, force residuals."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader
from .utils import FORCE_AU_TO_EV_ANG


def collect_forces(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset.graphs, batch_size=16, shuffle=False)
    pred_all, ref_all, elem_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            if device.type != 'cuda':
                batch = batch.to(device)
            forces = model(batch)
            pred_all.append(forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            ref_all.append(batch.dft_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            elem_all.append(batch.element_indices.cpu().numpy())
    return np.concatenate(pred_all), np.concatenate(ref_all), np.concatenate(elem_all)


def plot_training(train_hist, val_hist, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_hist)+1), train_hist, 'b-', label='Train', alpha=0.8)
    if val_hist:
        ax.plot(range(1, len(val_hist)+1), val_hist, 'r-', label='Val', alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Force RMSE (eV/A)')
    ax.set_title(f'Best val = {min(val_hist):.4f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(filepath, dpi=150); plt.close()


def plot_force_parity(model, dataset, device, filepath):
    pred, ref, _ = collect_forces(model, dataset, device)
    rmse = np.sqrt(np.mean((pred.ravel() - ref.ravel())**2))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(ref.ravel(), pred.ravel(), gridsize=80, mincnt=1, cmap='viridis', norm=LogNorm())
    lim = np.percentile(np.abs(np.concatenate([ref.ravel(), pred.ravel()])), 99.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('DFT (eV/A)'); ax.set_ylabel('Predicted (eV/A)')
    ax.set_title(f'RMSE = {rmse:.4f} eV/A'); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig(filepath, dpi=150); plt.close()


def generate_all_plots(model, dataset, device, train_hist=None, val_hist=None, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    if train_hist:
        plot_training(train_hist, val_hist or [], os.path.join(output_dir, 'training_curves.png'))
    try:
        plot_force_parity(model, dataset, device, os.path.join(output_dir, 'force_parity.png'))
    except Exception as e:
        print(f'  Force parity plot failed: {e}')
    print(f'  Plots: {output_dir}/')
