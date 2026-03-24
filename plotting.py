"""Diagnostic plots for GNN-Morse fitter (all units: eV/A, eV, Angstroms)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG,
    canonical_pair,
)


def _collect_forces(model, dataset, device, max_frames=None):
    """Run inference, return {pred_forces, dft_forces, elem_indices, frame_indices}."""
    model.eval()
    graphs = dataset.graphs[:max_frames] if max_frames else dataset.graphs
    loader = DataLoader(graphs, batch_size=32, shuffle=False)

    pred_list, ref_list, elem_list = [], [], []
    frame_list = []
    frame_offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            forces, _ = model(batch)

            pred_np = forces.cpu().numpy() * FORCE_AU_TO_EV_ANG
            ref_np = batch.dft_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG
            elem_np = batch.element_indices.cpu().numpy()

            # Frame indices per atom
            if hasattr(batch, 'batch') and batch.batch is not None:
                batch_idx = batch.batch.cpu().numpy()
            else:
                batch_idx = np.zeros(len(elem_np), dtype=int)
            frame_list.append(batch_idx + frame_offset)
            frame_offset = frame_list[-1].max() + 1

            pred_list.append(pred_np)
            ref_list.append(ref_np)
            elem_list.append(elem_np)

    return {
        'pred_forces': np.concatenate(pred_list),
        'dft_forces': np.concatenate(ref_list),
        'elem_indices': np.concatenate(elem_list),
        'frame_indices': np.concatenate(frame_list),
    }


def _collect_edge_data(model, dataset, device, max_frames=None):
    """Collect per-edge params {distances, pair_indices, D_e, alpha, r0, base_*}."""
    model.eval()
    graphs = dataset.graphs[:max_frames] if max_frames else dataset.graphs

    dist_list, pair_list = [], []
    De_list, alpha_list, r0_list = [], [], []

    with torch.no_grad():
        for graph in graphs:
            graph_dev = graph.to(device)
            params = model.get_per_edge_params(graph_dev)
            dist_list.append(graph.distances.numpy() * BOHR_TO_ANGSTROM)
            pair_list.append(graph.pair_indices.numpy())
            De_list.append(params['D_e'])
            alpha_list.append(params['alpha'])
            r0_list.append(params['r0'])

    base = model.get_base_params_display()

    return {
        'distances': np.concatenate(dist_list),
        'pair_indices': np.concatenate(pair_list),
        'D_e': np.concatenate(De_list),
        'alpha': np.concatenate(alpha_list),
        'r0': np.concatenate(r0_list),
        'base_D_e': base['D_e'],
        'base_alpha': base['alpha'],
        'base_r0': base['r0'],
    }



def plot_training(rmse_history, energy_history, val_history, filepath):
    """Training/validation loss curves (linear + log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = np.arange(1, len(rmse_history) + 1)

    # Linear scale
    ax = axes[0]
    ax.plot(epochs, rmse_history, 'b-', label='Train', alpha=0.8, lw=1.2)
    if val_history:
        ax.plot(epochs, val_history, 'r-', label='Val', alpha=0.8, lw=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.plot(epochs, rmse_history, 'b-', label='Train', alpha=0.8, lw=1.2)
    if val_history:
        ax.plot(epochs, val_history, 'r-', label='Val', alpha=0.8, lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Training Curves (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Add final values annotation
    final_train = rmse_history[-1] if rmse_history else 0
    final_val = val_history[-1] if val_history else 0
    best_val = min(val_history) if val_history else 0
    axes[0].annotate(
        f'Final: train={final_train:.4f}, val={final_val:.4f}\nBest val={best_val:.4f}',
        xy=(0.98, 0.98), xycoords='axes fraction',
        ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_force_parity(model, dataset, device, filepath, core_elements=None,
                      max_frames=None):
    """Parity plot with marginal histograms, showing core-atom and all-atom RMSE."""
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']

    elements = dataset.elements

    # Build core mask
    if core_elements:
        core_set = set(core_elements)
        core_atom_mask = np.array([elements[ei] in core_set for ei in elem_idx])
    else:
        core_atom_mask = np.ones(len(elem_idx), dtype=bool)

    # Flatten to components
    pred_flat = pred.ravel()
    ref_flat = ref.ravel()
    core_mask_3d = np.repeat(core_atom_mask, 3)

    pred_core = pred_flat[core_mask_3d]
    ref_core = ref_flat[core_mask_3d]

    rmse_all = np.sqrt(np.mean((pred_flat - ref_flat) ** 2))
    rmse_core = np.sqrt(np.mean((pred_core - ref_core) ** 2))

    # Figure with marginal histograms
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :3])
    ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    # Main hexbin — core atoms only
    hb = ax_main.hexbin(ref_core, pred_core, gridsize=80, mincnt=1,
                        cmap='viridis', norm=LogNorm())
    lim = np.percentile(np.abs(np.concatenate([ref_core, pred_core])), 99.5) * 1.1
    ax_main.plot([-lim, lim], [-lim, lim], 'r--', lw=1, alpha=0.7)
    ax_main.set_xlim(-lim, lim)
    ax_main.set_ylim(-lim, lim)
    ax_main.set_xlabel('DFT Force (eV/\u00c5)')
    ax_main.set_ylabel('Predicted Force (eV/\u00c5)')
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.15)

    # Colorbar
    cb = fig.colorbar(hb, ax=ax_right, fraction=0.8, pad=0.15, shrink=0.6)
    cb.set_label('Count', fontsize=8)

    # Title with both RMSEs
    core_str = ', '.join(core_elements) if core_elements else 'all'
    ax_top.set_title(
        f'Force Parity  |  Core ({core_str}) RMSE = {rmse_core:.4f} eV/\u00c5'
        f'  |  All-atom RMSE = {rmse_all:.4f} eV/\u00c5',
        fontsize=10, pad=10)

    # Marginal histograms
    bins = np.linspace(-lim, lim, 120)
    ax_top.hist(ref_core, bins=bins, color='steelblue', alpha=0.7, density=True,
                label='DFT')
    ax_top.hist(pred_core, bins=bins, color='coral', alpha=0.5, density=True,
                label='Predicted')
    ax_top.legend(fontsize=7, loc='upper right')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel('Density', fontsize=8)

    ax_right.hist(pred_core, bins=bins, orientation='horizontal',
                  color='coral', alpha=0.5, density=True)
    ax_right.hist(ref_core, bins=bins, orientation='horizontal',
                  color='steelblue', alpha=0.3, density=True)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel('Density', fontsize=8)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return rmse_core, rmse_all



def plot_per_element_parity(model, dataset, device, filepath, max_frames=None):
    """Separate parity panels for each element."""
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']
    elements = dataset.elements

    n_elem = len(elements)
    cols = min(n_elem, 4)
    rows = (n_elem + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows),
                             squeeze=False)

    for ei, elem in enumerate(elements):
        ax = axes[ei // cols][ei % cols]
        mask = elem_idx == ei
        if mask.sum() == 0:
            ax.set_title(f'{elem} (no atoms)')
            ax.set_visible(False)
            continue

        p = pred[mask].ravel()
        r = ref[mask].ravel()
        rmse = np.sqrt(np.mean((p - r) ** 2))
        mae = np.mean(np.abs(p - r))

        ax.hexbin(r, p, gridsize=50, mincnt=1, cmap='viridis', norm=LogNorm())
        lim = np.percentile(np.abs(np.concatenate([r, p])), 99) * 1.2
        if lim < 0.01:
            lim = 1.0
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, alpha=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{elem}  (RMSE={rmse:.4f}, MAE={mae:.4f} eV/\u00c5)',
                     fontsize=9)
        ax.set_xlabel('DFT (eV/\u00c5)', fontsize=8)
        ax.set_ylabel('Pred (eV/\u00c5)', fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)

        # Atom count annotation
        ax.annotate(f'N={mask.sum()}', xy=(0.03, 0.97), xycoords='axes fraction',
                    ha='left', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide empty subplots
    for i in range(n_elem, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('Per-Element Force Parity', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_force_residuals(model, dataset, device, filepath, max_frames=None):
    """Per-atom squared force residual |F_DFT - F_pred|^2 distribution, by element."""
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']
    elements = dataset.elements

    # Per-atom |F_err|^2 = sum over xyz of (F_pred - F_dft)^2
    sq_residual = np.sum((pred - ref) ** 2, axis=1)  # (N_atoms,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of |F_err|^2 by element (stacked)
    ax = axes[0]
    max_val = np.percentile(sq_residual, 99)
    bins = np.linspace(0, max_val, 80)
    elem_colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))

    for ei, elem in enumerate(elements):
        mask = elem_idx == ei
        if mask.sum() == 0:
            continue
        vals = sq_residual[mask]
        ax.hist(vals, bins=bins, alpha=0.6, label=f'{elem} (med={np.median(vals):.4f})',
                color=elem_colors[ei], edgecolor='none')

    ax.set_xlabel('|F$_{DFT}$ - F$_{pred}$|$^2$ (eV/\u00c5)$^2$')
    ax.set_ylabel('Count')
    ax.set_title('Force Residual Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: per-atom RMSE (sqrt of mean |F_err|^2 per atom across frames)
    ax = axes[1]

    frame_idx = data['frame_indices']
    n_atoms_per_frame = len(dataset.frame_data[0][0])

    # Local atom index = position within each frame
    unique_frames, first_indices = np.unique(frame_idx, return_index=True)
    frame_start = np.zeros(int(frame_idx.max()) + 1, dtype=int)
    frame_start[unique_frames] = first_indices
    local_atom_idx = np.arange(len(frame_idx)) - frame_start[frame_idx]

    # Accumulate per-atom |F_err|^2 with vectorized scatter
    valid = local_atom_idx < n_atoms_per_frame
    atom_sq_err = np.zeros(n_atoms_per_frame)
    atom_count = np.zeros(n_atoms_per_frame)
    np.add.at(atom_sq_err, local_atom_idx[valid], sq_residual[valid])
    np.add.at(atom_count, local_atom_idx[valid], 1)

    atom_count[atom_count == 0] = 1
    atom_rmse = np.sqrt(atom_sq_err / atom_count)

    # Color by element (use first frame's element assignment)
    symbols = dataset.frame_data[0][0]
    elem_for_atom = [dataset.element_to_index[s] for s in symbols]

    scatter_colors = [elem_colors[ei] for ei in elem_for_atom[:n_atoms_per_frame]]
    ax.bar(range(n_atoms_per_frame), atom_rmse, color=scatter_colors, width=1.0,
           edgecolor='none', alpha=0.8)
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('RMSE (eV/\u00c5)')
    ax.set_title('Per-Atom Force RMSE (averaged over frames)')
    ax.grid(True, alpha=0.2, axis='y')

    # Add element legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=elem_colors[ei], label=elem)
                      for ei, elem in enumerate(elements)]
    ax.legend(handles=legend_patches, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_pair_type_errors(model, dataset, device, filepath, max_frames=None):
    """Force RMSE contribution by pair type."""
    from torch_scatter import scatter_add

    model.eval()
    graphs = dataset.graphs[:max_frames] if max_frames else dataset.graphs
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    pair_names = dataset.pair_names

    # Per-pair-type force error accumulation
    n_pairs = len(pair_names)
    pair_sq_err = np.zeros(n_pairs)
    pair_count = np.zeros(n_pairs)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            forces_pred, _ = model(batch)
            force_err = (forces_pred - batch.dft_forces)  # (N, 3) AU

            # For each edge, compute the force contribution error
            # We attribute the force error at atom i to its edges
            edge_src = batch.edge_index[0]
            elem_idx = batch.element_indices
            pi = batch.pair_indices.cpu().numpy()

            # Per-atom squared error magnitude
            atom_sq_err = torch.sum(force_err ** 2, dim=1).cpu().numpy()  # (N,)

            # Attribute atom error to pair types of its edges (vectorized)
            src_np = edge_src.cpu().numpy()
            edge_atom_err = atom_sq_err[src_np]
            np.add.at(pair_sq_err, pi, edge_atom_err)
            np.add.at(pair_count, pi, 1)

    # Normalize
    pair_count[pair_count == 0] = 1
    pair_rmse = np.sqrt(pair_sq_err / pair_count) * FORCE_AU_TO_EV_ANG

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of RMSE by pair type
    ax = axes[0]
    colors = ['#e74c3c' if pair_rmse[i] > np.median(pair_rmse) else '#3498db'
              for i in range(n_pairs)]
    bars = ax.bar(range(n_pairs), pair_rmse, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Force Error by Pair Type')
    ax.grid(True, alpha=0.2, axis='y')

    # Annotate with edge counts
    for i, (bar, count) in enumerate(zip(bars, pair_count)):
        ax.annotate(f'{int(count)}', xy=(bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='bottom', fontsize=6)

    # Pie chart of error contribution
    ax = axes[1]
    total_err = pair_sq_err.sum()
    if total_err > 0:
        contributions = pair_sq_err / total_err * 100
        # Only show pairs with >1% contribution
        sig_mask = contributions > 1.0
        labels = [f'{pn}\n{c:.1f}%' for pn, c, m in
                  zip(pair_names, contributions, sig_mask) if m]
        sizes = contributions[sig_mask]
        other = contributions[~sig_mask].sum()
        if other > 0:
            labels.append(f'Other\n{other:.1f}%')
            sizes = np.append(sizes, other)
        ax.pie(sizes, labels=labels, autopct='', startangle=90,
               textprops={'fontsize': 8})
        ax.set_title('Error Contribution by Pair Type')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_gnn_corrections(model, dataset, device, filepath, max_frames=50):
    """Visualize how much the GNN shifts parameters from base values."""
    if not model.use_gnn:
        print(f"Skipping GNN correction plot (GNN disabled)")
        return

    edge_data = _collect_edge_data(model, dataset, device, max_frames)
    pair_names = dataset.pair_names

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    param_labels = [
        ('D_e', 'base_D_e', 'D$_e$ (eV)'),
        ('alpha', 'base_alpha', r'$\alpha$ (1/\u00c5)'),
        ('r0', 'base_r0', 'r$_0$ (\u00c5)'),
    ]

    n_pairs = len(pair_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    for col, (key, base_key, label) in enumerate(param_labels):
        ax = axes[col]
        edge_vals = edge_data[key]
        base_vals = edge_data[base_key]
        pair_idx = edge_data['pair_indices']

        for pi, pn in enumerate(pair_names):
            mask = pair_idx == pi
            if mask.sum() == 0:
                continue
            vals = edge_vals[mask]
            base = base_vals[pi]
            deltas = vals - base

            # Violin-like: show distribution of corrections
            parts = ax.violinplot([deltas], positions=[pi], showmeans=True,
                                  showmedians=True, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[pi])
                pc.set_alpha(0.6)

        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f'\u0394{label}')
        ax.set_title(f'GNN Correction: {label}')
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('GNN Parameter Corrections (per-edge value minus base)', fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_embeddings(model, dataset, device, filepath):
    """PCA of GNN per-atom embeddings, colored by element and annotated with clusters."""
    if not model.use_gnn:
        print(f"Skipping embedding plot (GNN disabled)")
        return

    from sklearn.decomposition import PCA

    graph = dataset.graphs[0].to(device)
    embeddings = model.get_embeddings(graph)
    if embeddings is None:
        return

    symbols = dataset.frame_data[0][0]
    elements = dataset.elements
    elem_idx = [dataset.element_to_index[s] for s in symbols]

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by element
    ax = axes[0]
    elem_colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))
    for ei, elem in enumerate(elements):
        mask = np.array(elem_idx) == ei
        if mask.sum() == 0:
            continue
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[elem_colors[ei]],
                   label=f'{elem} ({mask.sum()})', s=20, alpha=0.7, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('GNN Embeddings by Element')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: inorganic only, with k-means clustering overlay
    ax = axes[1]
    inorganic = set(dataset.knn_config.get('inorganic_elements', ['Cd', 'Se']))
    inorg_mask = np.array([s in inorganic for s in symbols])

    if inorg_mask.sum() > 0:
        from sklearn.cluster import KMeans

        inorg_emb = embeddings[inorg_mask]
        inorg_2d = emb_2d[inorg_mask]
        inorg_symbols = [s for s, m in zip(symbols, inorg_mask) if m]

        # Cluster each element separately (matching export logic)
        cluster_labels = np.zeros(inorg_mask.sum(), dtype=int)
        label_offset = 0
        elem_cluster_info = {}

        for elem in sorted(inorganic):
            e_mask = np.array(inorg_symbols) == elem
            if e_mask.sum() < 3:
                cluster_labels[e_mask] = label_offset
                elem_cluster_info[elem] = 1
                label_offset += 1
                continue

            # Same clustering as lammps_export
            from sklearn.metrics import silhouette_score
            e_emb = inorg_emb[e_mask]
            best_k, best_score = 2, -1
            for k in range(2, min(6, e_mask.sum())):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbls = km.fit_predict(e_emb)
                if len(set(lbls)) >= 2:
                    score = silhouette_score(e_emb, lbls)
                    if score > best_score:
                        best_score = score
                        best_k = k

            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            lbls = km.fit_predict(e_emb)
            cluster_labels[e_mask] = lbls + label_offset
            elem_cluster_info[elem] = best_k
            label_offset += best_k

        # Plot with cluster coloring
        unique_labels = np.unique(cluster_labels)
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for li, label in enumerate(unique_labels):
            mask = cluster_labels == label
            elem = inorg_symbols[np.where(mask)[0][0]]
            count = mask.sum()
            ax.scatter(inorg_2d[mask, 0], inorg_2d[mask, 1],
                       c=[cluster_colors[li]], s=25, alpha=0.7,
                       label=f'{elem}_{chr(65 + label - sum(elem_cluster_info[e] for e in sorted(inorganic) if e < elem))} ({count})',
                       edgecolors='gray', linewidths=0.3)

        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.set_title(f'Inorganic Atom Clusters (k-means on embeddings)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle('GNN Per-Atom Embeddings (PCA, frame 0)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_morse_curves(model, dataset, device, filepath,
                      frozen_pairs=None, max_frames=50):
    """Plot Morse V(r) curves showing per-edge parameter spread vs base."""
    edge_data = _collect_edge_data(model, dataset, device, max_frames)
    pair_names = dataset.pair_names
    frozen = set(frozen_pairs or [])

    # Only plot non-frozen pairs
    active_pairs = [(pi, pn) for pi, pn in enumerate(pair_names) if pn not in frozen]
    n_active = len(active_pairs)
    if n_active == 0:
        return

    cols = min(n_active, 4)
    rows = (n_active + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                             squeeze=False)

    r_plot = np.linspace(0.5, 6.0, 500)

    for idx, (pi, pn) in enumerate(active_pairs):
        ax = axes[idx // cols][idx % cols]
        mask = edge_data['pair_indices'] == pi
        if mask.sum() == 0:
            ax.set_title(f'{pn} (no edges)')
            continue

        # Base Morse curve
        D_base = edge_data['base_D_e'][pi]
        a_base = edge_data['base_alpha'][pi]
        r0_base = edge_data['base_r0'][pi]
        x_base = r_plot - r0_base
        V_base = D_base * (1 - np.exp(-a_base * x_base)) ** 2
        ax.plot(r_plot, V_base, 'k--', lw=2, alpha=0.7, label='Base')

        # Sample of per-edge curves (show spread)
        D_edges = edge_data['D_e'][mask]
        a_edges = edge_data['alpha'][mask]
        r0_edges = edge_data['r0'][mask]

        # Percentile curves
        for pct, alpha, ls in [(10, 0.3, ':'), (50, 0.8, '-'), (90, 0.3, ':')]:
            D_p = np.percentile(D_edges, pct)
            a_p = np.percentile(a_edges, pct)
            r0_p = np.percentile(r0_edges, pct)
            x_p = r_plot - r0_p
            V_p = D_p * (1 - np.exp(-a_p * x_p)) ** 2
            label = f'P{pct}' if pct != 50 else 'Median'
            ax.plot(r_plot, V_p, color='steelblue', ls=ls, alpha=alpha,
                    lw=1.5, label=label)

        # Show actual edge distance distribution
        dists = edge_data['distances'][mask]
        ax2 = ax.twinx()
        ax2.hist(dists, bins=40, alpha=0.15, color='orange', density=True)
        ax2.set_ylabel('Edge dist. density', fontsize=7, color='orange')
        ax2.tick_params(axis='y', labelsize=6, colors='orange')

        ax.set_xlabel('r (\u00c5)', fontsize=8)
        ax.set_ylabel('V(r) (eV)', fontsize=8)
        ax.set_title(f'{pn}  ({mask.sum()} edges)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylim(bottom=-0.1)
        ax.set_ylim(top=min(D_base * 3, np.percentile(D_edges, 95) * 4, 5.0))
        ax.grid(True, alpha=0.15)

    for i in range(n_active, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('Morse Potential Curves (base vs GNN-corrected spread)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_edge_distances(dataset, filepath):
    """KNN edge distance distributions by pair type (frame 0)."""
    graph = dataset.graphs[0]
    pair_names = dataset.pair_names
    distances = graph.distances.numpy() * BOHR_TO_ANGSTROM
    pair_idx = graph.pair_indices.numpy()

    active = [(pi, pn) for pi, pn in enumerate(pair_names)
              if (pair_idx == pi).sum() > 0]
    n = len(active)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             squeeze=False)

    for idx, (pi, pn) in enumerate(active):
        ax = axes[idx // cols][idx % cols]
        mask = pair_idx == pi
        d = distances[mask]
        ax.hist(d, bins=40, color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(np.median(d), color='red', ls='--', lw=1,
                   label=f'median={np.median(d):.2f}\u00c5')
        ax.set_xlabel('Distance (\u00c5)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(f'{pn} ({mask.sum()//2} undirected)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('KNN Edge Distance Distributions (frame 0)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


# ── Master function ─────────────────────────────────────────────────

def generate_all_plots(model, dataset, device, config,
                       rmse_history=None, energy_history=None, val_history=None,
                       output_dir='.'):
    """Generate all diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)

    core_elements = config.get('core_elements', [])
    frozen_pairs = list(config.get('fixed_pairs', {}).keys())
    max_frames = len(dataset.graphs)  # use all frames

    print(f"\n{'='*60}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print('='*60)

    # 1. Training curves
    if rmse_history:
        plot_training(rmse_history, energy_history or [], val_history or [],
                      os.path.join(output_dir, 'training_curves.png'))

    # 2. Force parity (fixed!)
    plot_force_parity(model, dataset, device,
                      os.path.join(output_dir, 'force_parity.png'),
                      core_elements=core_elements, max_frames=max_frames)

    # 3. Per-element parity
    plot_per_element_parity(model, dataset, device,
                            os.path.join(output_dir, 'per_element_parity.png'),
                            max_frames=max_frames)

    # 4. Force residuals
    plot_force_residuals(model, dataset, device,
                         os.path.join(output_dir, 'force_residuals.png'),
                         max_frames=max_frames)

    # 5. Per-pair-type errors
    plot_pair_type_errors(model, dataset, device,
                          os.path.join(output_dir, 'pair_type_errors.png'),
                          max_frames=max_frames)

    # 6. GNN corrections
    plot_gnn_corrections(model, dataset, device,
                         os.path.join(output_dir, 'gnn_corrections.png'),
                         max_frames=min(50, max_frames))

    # 7. Embeddings
    plot_embeddings(model, dataset, device,
                    os.path.join(output_dir, 'embeddings_pca.png'))

    # 8. Morse curves
    plot_morse_curves(model, dataset, device,
                      os.path.join(output_dir, 'morse_curves.png'),
                      frozen_pairs=frozen_pairs,
                      max_frames=min(50, max_frames))

    # 9. Edge distances
    plot_edge_distances(dataset,
                        os.path.join(output_dir, 'edge_distances.png'))

    print(f"\nAll plots saved to {output_dir}/")
