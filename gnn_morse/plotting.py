import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG,
    canonical_pair,
)


def _collect_forces(model, dataset, device, max_frames=None):
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



def plot_training(train_rmse_history, val_rmse_history, filepath):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = np.arange(1, len(train_rmse_history) + 1)

    ax.plot(epochs, train_rmse_history, 'b-', label='Train', alpha=0.8, lw=1.2)
    if val_rmse_history:
        ax.plot(epochs, val_rmse_history, 'r-', label='Val', alpha=0.8, lw=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add final values annotation
    final_train = train_rmse_history[-1] if train_rmse_history else 0
    final_val = val_rmse_history[-1] if val_rmse_history else 0
    best_val = min(val_rmse_history) if val_rmse_history else 0
    ax.annotate(
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

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    hb = ax.hexbin(ref_core, pred_core, gridsize=80, mincnt=1,
                    cmap='viridis', norm=LogNorm())
    lim = np.percentile(np.abs(np.concatenate([ref_core, pred_core])), 99.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('DFT Force (eV/\u00c5)')
    ax.set_ylabel('Predicted Force (eV/\u00c5)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('Count', fontsize=8)

    core_str = ', '.join(core_elements) if core_elements else 'all'
    ax.set_title(
        f'Force Parity  |  Core ({core_str}) RMSE = {rmse_core:.4f} eV/\u00c5'
        f'  |  All-atom RMSE = {rmse_all:.4f} eV/\u00c5',
        fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return rmse_core, rmse_all



def plot_core_element_parity(model, dataset, device, filepath,
                             core_elements=None, max_frames=None):
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']
    elements = dataset.elements

    # Filter to core elements only
    if core_elements:
        plot_elems = [(ei, e) for ei, e in enumerate(elements) if e in set(core_elements)]
    else:
        plot_elems = list(enumerate(elements))

    n_elem = len(plot_elems)
    if n_elem == 0:
        return

    cols = min(n_elem, 4)
    rows = (n_elem + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows),
                             squeeze=False)

    for idx, (ei, elem) in enumerate(plot_elems):
        ax = axes[idx // cols][idx % cols]
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

    fig.suptitle('Core Element Force Parity', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_force_error_hist(model, dataset, device, filepath, max_frames=None):
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']
    elements = dataset.elements

    sq_residual = np.sum((pred - ref) ** 2, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

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
    ax.set_title('Force Error Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_per_atom_rmse(model, dataset, device, filepath,
                       core_elements=None, max_frames=None):
    data = _collect_forces(model, dataset, device, max_frames)
    pred = data['pred_forces']
    ref = data['dft_forces']
    elem_idx = data['elem_indices']
    frame_idx = data['frame_indices']
    elements = dataset.elements

    sq_residual = np.sum((pred - ref) ** 2, axis=1)
    n_atoms_per_frame = len(dataset.frame_data[0][0])

    # Local atom index within each frame
    unique_frames, first_indices = np.unique(frame_idx, return_index=True)
    frame_start = np.zeros(int(frame_idx.max()) + 1, dtype=int)
    frame_start[unique_frames] = first_indices
    local_atom_idx = np.arange(len(frame_idx)) - frame_start[frame_idx]

    # Per-atom RMSE across frames
    valid = local_atom_idx < n_atoms_per_frame
    atom_sq_err = np.zeros(n_atoms_per_frame)
    atom_count = np.zeros(n_atoms_per_frame)
    np.add.at(atom_sq_err, local_atom_idx[valid], sq_residual[valid])
    np.add.at(atom_count, local_atom_idx[valid], 1)
    atom_count[atom_count == 0] = 1
    atom_rmse = np.sqrt(atom_sq_err / atom_count)

    symbols = dataset.frame_data[0][0]

    # Filter to core (inorganic) elements only
    if core_elements:
        core_set = set(core_elements)
    else:
        core_set = set(elements)
    core_mask = np.array([s in core_set for s in symbols])

    # Group atoms by element, sorted by RMSE descending within each group
    from matplotlib.patches import Patch
    elem_colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))
    elem_color_map = {e: elem_colors[i] for i, e in enumerate(elements)}

    groups = []
    for elem in sorted(core_set):
        atom_indices = [i for i, s in enumerate(symbols) if s == elem and core_mask[i]]
        if not atom_indices:
            continue
        rmses = atom_rmse[atom_indices]
        order = np.argsort(-rmses)
        groups.append((elem, np.array(atom_indices)[order], rmses[order]))

    if not groups:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x_pos = 0
    tick_positions = []
    tick_labels = []
    separator_positions = []

    for gi, (elem, indices, rmses) in enumerate(groups):
        xs = np.arange(x_pos, x_pos + len(indices))
        color = elem_color_map.get(elem, 'gray')
        ax.bar(xs, rmses, color=color, width=1.0, edgecolor='none', alpha=0.8)
        tick_positions.append(x_pos + len(indices) / 2)
        tick_labels.append(f'{elem} ({len(indices)})')
        x_pos += len(indices)
        if gi < len(groups) - 1:
            separator_positions.append(x_pos - 0.5)
            x_pos += 2  # gap between groups

    for sp in separator_positions:
        ax.axvline(sp, color='gray', ls='-', lw=0.5, alpha=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel('RMSE (eV/\u00c5)')
    ax.set_title('Per-Atom Force RMSE (grouped by element, sorted descending)')
    ax.grid(True, alpha=0.2, axis='y')

    legend_patches = [Patch(facecolor=elem_color_map.get(e, 'gray'), label=e)
                      for e in sorted(core_set) if e in elem_color_map]
    ax.legend(handles=legend_patches, fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_pair_type_errors(model, dataset, device, filepath, max_frames=None):
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

            edge_src = batch.edge_index[0]
            pi = batch.pair_indices.cpu().numpy()

            atom_sq_err = torch.sum(force_err ** 2, dim=1).cpu().numpy()

            src_np = edge_src.cpu().numpy()
            edge_atom_err = atom_sq_err[src_np]
            np.add.at(pair_sq_err, pi, edge_atom_err)
            np.add.at(pair_count, pi, 1)

    # Normalize
    pair_count[pair_count == 0] = 1
    pair_rmse = np.sqrt(pair_sq_err / pair_count) * FORCE_AU_TO_EV_ANG

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: RMSE bar chart (unchanged)
    ax = axes[0]
    colors = ['#e74c3c' if pair_rmse[i] > np.median(pair_rmse) else '#3498db'
              for i in range(n_pairs)]
    bars = ax.bar(range(n_pairs), pair_rmse, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Force RMSE (eV/\u00c5)')
    ax.set_title('Force Error by Pair Type')
    ax.grid(True, alpha=0.2, axis='y')

    for i, (bar, count) in enumerate(zip(bars, pair_count)):
        ax.annotate(f'{int(count)}', xy=(bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='bottom', fontsize=6)

    # Right: horizontal bar chart of % error contribution (replaces pie)
    ax = axes[1]
    total_err = pair_sq_err.sum()
    if total_err > 0:
        contributions = pair_sq_err / total_err * 100
        # Sort descending
        order = np.argsort(contributions)[::-1]
        sorted_names = [pair_names[i] for i in order]
        sorted_contribs = contributions[order]

        bar_colors = ['#e74c3c' if c > 100 / n_pairs else '#3498db'
                      for c in sorted_contribs]
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_contribs, color=bar_colors, edgecolor='white', lw=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel('Error Contribution (%)')
        ax.set_title('Error Contribution by Pair Type')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x')

        # Label bars with percentages
        for i, (val, name) in enumerate(zip(sorted_contribs, sorted_names)):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_gnn_corrections(model, dataset, device, filepath, max_frames=50):
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

        box_data = []
        box_positions = []
        box_colors = []

        for pi, pn in enumerate(pair_names):
            mask = pair_idx == pi
            if mask.sum() == 0:
                continue
            vals = edge_vals[mask]
            base = base_vals[pi]
            deltas = vals - base
            box_data.append(deltas)
            box_positions.append(pi)
            box_colors.append(colors[pi])

        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', markersize=2, alpha=0.3))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

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



def plot_morse_curves(model, dataset, device, filepath,
                      frozen_pairs=None, max_frames=50):
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
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows),
                             squeeze=False)

    r_plot = np.linspace(0.5, 6.0, 500)

    for idx, (pi, pn) in enumerate(active_pairs):
        ax = axes[idx // cols][idx % cols]
        mask = edge_data['pair_indices'] == pi
        if mask.sum() == 0:
            ax.set_title(f'{pn} (no edges)', fontsize=11)
            continue

        # Base Morse curve
        D_base = edge_data['base_D_e'][pi]
        a_base = edge_data['base_alpha'][pi]
        r0_base = edge_data['base_r0'][pi]
        x_base = r_plot - r0_base
        V_base = D_base * (1 - np.exp(-a_base * x_base)) ** 2
        ax.plot(r_plot, V_base, 'k--', lw=2, alpha=0.7, label='Base')

        # Per-edge parameter arrays
        D_edges = edge_data['D_e'][mask]
        a_edges = edge_data['alpha'][mask]
        r0_edges = edge_data['r0'][mask]

        # Median curve
        D_med = np.median(D_edges)
        a_med = np.median(a_edges)
        r0_med = np.median(r0_edges)
        x_med = r_plot - r0_med
        V_med = D_med * (1 - np.exp(-a_med * x_med)) ** 2
        ax.plot(r_plot, V_med, color='steelblue', ls='-', lw=1.8, label='Median')

        # 10th-90th percentile shaded band
        D_10, D_90 = np.percentile(D_edges, 10), np.percentile(D_edges, 90)
        a_10, a_90 = np.percentile(a_edges, 10), np.percentile(a_edges, 90)
        r0_10, r0_90 = np.percentile(r0_edges, 10), np.percentile(r0_edges, 90)

        V_lo = D_10 * (1 - np.exp(-a_10 * (r_plot - r0_10))) ** 2
        V_hi = D_90 * (1 - np.exp(-a_90 * (r_plot - r0_90))) ** 2
        ax.fill_between(r_plot, V_lo, V_hi, color='steelblue', alpha=0.15,
                         label='10th\u201390th pctile')

        # Distance histogram overlay
        dists = edge_data['distances'][mask]
        ax2 = ax.twinx()
        ax2.hist(dists, bins=40, alpha=0.15, color='orange', density=True)
        ax2.set_ylabel('Edge dist. density', fontsize=8, color='orange')
        ax2.tick_params(axis='y', labelsize=7, colors='orange')

        ax.set_xlabel('r (\u00c5)', fontsize=10)
        ax.set_ylabel('V(r) (eV)', fontsize=10)
        ax.set_title(f'{pn}  ({mask.sum()} edges)', fontsize=11)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=-0.1)
        ax.set_ylim(top=max(2 * D_med, 3.0))
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=8)

    for i in range(n_active, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('Morse Potential Curves (base vs GNN-corrected spread)', fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")



def plot_edge_distances(dataset, filepath, gap_cuts=None):
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

        # Add gap cutoff vertical line if available
        if gap_cuts and pn in gap_cuts:
            cutoff = gap_cuts[pn]
            ax.axvline(cutoff, color='darkred', ls=':', lw=1.5,
                       label=f'cutoff={cutoff:.2f}\u00c5')

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
                       train_rmse_history=None, val_rmse_history=None,
                       output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)

    core_elements = config.get('core_elements', [])
    frozen_pairs = list(config.get('fixed_pairs', {}).keys())
    max_frames = len(dataset.graphs)  # use all frames

    print(f"\n{'='*60}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print('='*60)

    # 1. Training curves (single panel)
    if train_rmse_history:
        plot_training(train_rmse_history, val_rmse_history or [],
                      os.path.join(output_dir, 'training_curves.png'))

    # 2. Force parity
    plot_force_parity(model, dataset, device,
                      os.path.join(output_dir, 'force_parity.png'),
                      core_elements=core_elements, max_frames=max_frames)

    # 3. Core element parity
    plot_core_element_parity(model, dataset, device,
                             os.path.join(output_dir, 'core_element_parity.png'),
                             core_elements=core_elements, max_frames=max_frames)

    # 4a. Force error histogram
    plot_force_error_hist(model, dataset, device,
                          os.path.join(output_dir, 'force_error_distribution.png'),
                          max_frames=max_frames)

    # 4b. Per-atom RMSE (core elements only)
    plot_per_atom_rmse(model, dataset, device,
                       os.path.join(output_dir, 'per_atom_rmse.png'),
                       core_elements=core_elements, max_frames=max_frames)

    # 5. Per-pair-type errors (bar chart replaces pie)
    plot_pair_type_errors(model, dataset, device,
                          os.path.join(output_dir, 'pair_type_errors.png'),
                          max_frames=max_frames)

    # 6. GNN corrections (box plots)
    plot_gnn_corrections(model, dataset, device,
                         os.path.join(output_dir, 'gnn_corrections.png'),
                         max_frames=min(50, max_frames))

    # 7. Morse curves (shaded bands)
    plot_morse_curves(model, dataset, device,
                      os.path.join(output_dir, 'morse_curves.png'),
                      frozen_pairs=frozen_pairs,
                      max_frames=min(50, max_frames))

    # 8. Edge distances (with cutoff lines)
    gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
    plot_edge_distances(dataset,
                        os.path.join(output_dir, 'edge_distances.png'),
                        gap_cuts=gap_cuts)

    print(f"\nAll plots saved to {output_dir}/")
