import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader

from .utils import BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG


# Colors: Okabe-Ito then Tol muted, all colorblind-friendly
_PALETTE = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7', '#000000',
    '#CC6677', '#332288', '#DDCC77', '#117733',
    '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499',
]


def _collect_forces(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset.graphs, batch_size=32, shuffle=False)

    pred_all, ref_all, elem_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            forces, _ = model(batch)
            pred_all.append(forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            ref_all.append(batch.dft_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            elem_all.append(batch.element_indices.cpu().numpy())

    return (np.concatenate(pred_all),
            np.concatenate(ref_all),
            np.concatenate(elem_all))


def plot_training(train_history, val_history, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_history) + 1)
    ax.plot(epochs, train_history, 'b-', label='Train', alpha=0.8, lw=1.2)
    if val_history:
        ax.plot(epochs, val_history, 'r-', label='Val', alpha=0.8, lw=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_val = min(val_history) if val_history else 0
    ax.annotate(
        f'Final: train={train_history[-1]:.4f}, val={val_history[-1]:.4f}\n'
        f'Best val={best_val:.4f}',
        xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_force_parity(model, dataset, device, filepath, core_elements=None):
    pred, ref, elem_idx = _collect_forces(model, dataset, device)
    elements = dataset.elements

    if core_elements:
        core_mask = np.array([elements[ei] in set(core_elements) for ei in elem_idx])
    else:
        core_mask = np.ones(len(elem_idx), dtype=bool)

    # Flatten xyz components, filter to core atoms
    core_3d = np.repeat(core_mask, 3)
    pred_core = pred.ravel()[core_3d]
    ref_core = ref.ravel()[core_3d]

    rmse_all = np.sqrt(np.mean((pred.ravel() - ref.ravel()) ** 2))
    rmse_core = np.sqrt(np.mean((pred_core - ref_core) ** 2))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(ref_core, pred_core, gridsize=80, mincnt=1, cmap='viridis', norm=LogNorm())
    lim = np.percentile(np.abs(np.concatenate([ref_core, pred_core])), 99.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('DFT Force (eV/\u00c5)')
    ax.set_ylabel('Predicted Force (eV/\u00c5)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.8, label='Count')

    core_str = ', '.join(core_elements) if core_elements else 'all'
    ax.set_title(f'Core ({core_str}) RMSE = {rmse_core:.4f}  |  All RMSE = {rmse_all:.4f} eV/\u00c5',
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_force_residuals(model, dataset, device, filepath):
    pred, ref, elem_idx = _collect_forces(model, dataset, device)
    elements = dataset.elements
    sq_err = np.sum((pred - ref) ** 2, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(sq_err, 99), 80)
    colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))

    for ei, elem in enumerate(elements):
        mask = elem_idx == ei
        if not mask.any():
            continue
        vals = sq_err[mask]
        ax.hist(vals, bins=bins, alpha=0.6, color=colors[ei], edgecolor='none',
                label=f'{elem} (med={np.median(vals):.4f})')

    ax.set_xlabel('|F_DFT - F_pred|^2 (eV/\u00c5)^2')
    ax.set_ylabel('Count')
    ax.set_title('Force Error Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def _structural_labels(symbols, edge_index_np, inorganic_elements):
    inorganic = set(inorganic_elements)
    n_atoms = len(symbols)
    src, tgt = edge_index_np

    # Count cross-species inorganic neighbors per atom
    cross_coord = np.zeros(n_atoms, dtype=int)
    for s, t in zip(src, tgt):
        if symbols[s] in inorganic and symbols[t] in inorganic and symbols[s] != symbols[t]:
            cross_coord[s] += 1

    # Max coordination per element
    elem_max = {}
    for i, sym in enumerate(symbols):
        if sym in inorganic:
            elem_max[sym] = max(elem_max.get(sym, 0), cross_coord[i])

    labels = []
    for i, sym in enumerate(symbols):
        if sym not in inorganic:
            labels.append(sym)
        elif cross_coord[i] >= elem_max.get(sym, 0):
            labels.append(f"{sym}_core")
        else:
            labels.append(f"{sym}_surf")
    return labels


def plot_umap(model, dataset, device, filepath, stride=10, core_elements=None):
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed, skipping UMAP")
        return

    model.eval()
    inorganic = core_elements or model.elements

    all_embeddings, all_labels = [], []
    for fi in range(0, len(dataset.graphs), stride):
        graph = dataset.graphs[fi].to(device)
        emb = model.get_embeddings(graph)
        if emb is None:
            print("  No GNN embeddings, skipping UMAP")
            return
        all_embeddings.append(emb)
        symbols = list(dataset.frame_data[fi][0])
        all_labels.extend(_structural_labels(symbols, graph.edge_index.cpu().numpy(), inorganic))

    embeddings = np.concatenate(all_embeddings)
    n_frames = len(range(0, len(dataset.graphs), stride))
    print(f"  UMAP: {embeddings.shape[0]} points ({n_frames} frames)")

    coords = UMAP(n_components=2, random_state=42, n_neighbors=15,
                   min_dist=0.1, n_jobs=1).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(all_labels))
    for i, label in enumerate(unique_labels):
        mask = [j for j, l in enumerate(all_labels) if l == label]
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=label,
                   s=12, alpha=0.5, edgecolors='none')

    ax.legend(fontsize=8, ncol=2, markerscale=2.5, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'GNN Atom Embeddings ({n_frames} frames)')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_all_plots(model, dataset, device, config,
                       train_rmse_history=None, val_rmse_history=None,
                       output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    core_elements = config.get('core_elements', [])

    print(f"\nGenerating plots...")

    if train_rmse_history:
        plot_training(train_rmse_history, val_rmse_history or [],
                      os.path.join(output_dir, 'training_curves.png'))

    plot_force_parity(model, dataset, device,
                      os.path.join(output_dir, 'force_parity.png'),
                      core_elements=core_elements)

    plot_force_residuals(model, dataset, device,
                         os.path.join(output_dir, 'force_error_distribution.png'))

    plot_umap(model, dataset, device,
              os.path.join(output_dir, 'umap_embeddings.png'),
              stride=10, core_elements=core_elements or None)

    print(f"All plots saved to {output_dir}/")
