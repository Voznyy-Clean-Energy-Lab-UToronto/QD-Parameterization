import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader

from .utils import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, FORCE_AU_TO_EV_ANG
from .labelling import classify_atoms, _compute_cross_species_cn


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


def plot_force_residuals(model, dataset, device, filepath,
                         inorganic_elements=None):
    pred, ref, elem_idx = _collect_forces(model, dataset, device)
    elements = dataset.elements
    sq_err = np.sum((pred - ref) ** 2, axis=1)

    # Filter to inorganic elements only if specified
    if inorganic_elements:
        show_set = set(inorganic_elements)
    else:
        show_set = set(elements)
    show_indices = [ei for ei, e in enumerate(elements) if e in show_set]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(sq_err, 99), 80)
    colors = plt.cm.Set1(np.linspace(0, 1, len(show_indices)))

    for ci, ei in enumerate(show_indices):
        elem = elements[ei]
        mask = elem_idx == ei
        if not mask.any():
            continue
        vals = sq_err[mask]
        ax.hist(vals, bins=bins, alpha=0.6, color=colors[ci], edgecolor='none',
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


def _get_physical_labels(model, dataset, device, config, stride=10):
    symbols = list(dataset.frame_data[0][0])
    inorganic = sorted(set(config.get('knn_edges', {}).get(
        'inorganic_elements', ['Cd', 'Se'])))

    cutoffs_bohr = getattr(dataset, 'cutoffs_bohr', None)
    if cutoffs_bohr is None:
        gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
        cutoffs_bohr = {pair: cut * ANGSTROM_TO_BOHR for pair, cut in gap_cuts.items()}

    # Compute per-atom CN (averaged over strided frames)
    n_atoms = len(symbols)
    cn_accum = np.zeros(n_atoms, dtype=np.float64)
    n_frames = 0
    for fi in range(0, len(dataset.frame_data), max(1, stride)):
        syms_f, pos_f = dataset.frame_data[fi]
        cn_accum += _compute_cross_species_cn(syms_f, pos_f, inorganic, cutoffs_bohr)
        n_frames += 1
    avg_cn = cn_accum / max(n_frames, 1)

    # Try VQ physical labels first
    model.eval()
    graph0 = dataset.graphs[0].to(device)
    vq_result = model.get_atom_types(graph0)

    if vq_result is not None:
        from .labelling import compute_physical_labels
        _, vq_type_names = vq_result
        physical_labels, _ = compute_physical_labels(
            symbols, dataset.frame_data[0][1], vq_type_names,
            inorganic, cutoffs_bohr,
            frame_data=dataset.frame_data, stride=stride)
        return physical_labels, avg_cn

    # Fallback: CN-based classification
    atom_labels, _, _ = classify_atoms(
        symbols, dataset.frame_data, cutoffs_bohr, inorganic,
        stride=stride,)
    return atom_labels, avg_cn


def plot_cn_vs_label(model, dataset, device, config, filepath, stride=10):
    inorganic = sorted(set(config.get('knn_edges', {}).get('inorganic_elements', ['Cd', 'Se'])))
    inorganic_set = set(inorganic)
    symbols_arr = np.array(dataset.frame_data[0][0])

    atom_labels, avg_cn = _get_physical_labels(model, dataset, device, config, stride)

    # Build per-element index map
    inorg_by_elem = {}
    for elem in sorted(inorganic_set):
        mask = symbols_arr == elem
        if mask.any():
            inorg_by_elem[elem] = np.where(mask)[0]
    inorg_elems = sorted(inorg_by_elem.keys())

    cn_rounded = np.round(avg_cn).astype(int)

    n_plots = len(inorg_elems)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes[0]

    for ax, elem in zip(axes, inorg_elems):
        idx = inorg_by_elem.get(elem, [])
        if len(idx) == 0:
            continue

        elem_cns = cn_rounded[idx]
        elem_labels = [atom_labels[i] for i in idx]
        unique_labels = sorted(set(elem_labels))
        unique_cns = sorted(set(elem_cns))

        label_counts = {label: [] for label in unique_labels}
        for cn_val in unique_cns:
            for label in unique_labels:
                count = sum(1 for i, ai in enumerate(idx)
                            if elem_cns[i] == cn_val and elem_labels[i] == label)
                label_counts[label].append(count)

        x = np.arange(len(unique_cns))
        width = 0.6
        bottom = np.zeros(len(unique_cns))
        for li, label in enumerate(unique_labels):
            color = _PALETTE[li % len(_PALETTE)]
            ax.bar(x, label_counts[label], width, bottom=bottom,
                   label=label, color=color, edgecolor='white', linewidth=0.5)
            bottom += np.array(label_counts[label])

        ax.set_xticks(x)
        ax.set_xticklabels([str(cn) for cn in unique_cns])
        ax.set_xlabel('Cross-species CN')
        ax.set_ylabel('Atom count')
        ax.set_title(f'{elem}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('VQ Subtype vs Coordination Number', fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_umap(model, dataset, device, filepath, stride=10, core_elements=None,
              config=None):
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed, skipping UMAP")
        return

    model.eval()

    # Use VQ-assigned labels if available, else element-only
    graph0 = dataset.graphs[0].to(device)
    vq_result = model.get_atom_types(graph0)
    if vq_result is not None:
        _, vq_type_names = vq_result
        atom_labels = vq_type_names
    else:
        symbols_list = list(dataset.frame_data[0][0])
        atom_labels = symbols_list

    # If VQ active and config available, compute physical labels
    if vq_result is not None and config is not None:
        try:
            from .labelling import compute_physical_labels
            symbols = list(dataset.frame_data[0][0])
            positions_bohr = dataset.frame_data[0][1]
            inorganic = sorted(set(config.get('knn_edges', {}).get(
                'inorganic_elements', ['Cd', 'Se'])))
            cutoffs_bohr = getattr(dataset, 'cutoffs_bohr', None)
            if cutoffs_bohr is None:
                gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
                cutoffs_bohr = {pair: cut * ANGSTROM_TO_BOHR
                                for pair, cut in gap_cuts.items()}
            physical_labels, _ = compute_physical_labels(
                symbols, positions_bohr, vq_type_names,
                inorganic, cutoffs_bohr,
                frame_data=dataset.frame_data, stride=10)
            atom_labels = physical_labels
        except Exception as e:
            print(f"  [Warning] Physical labels failed: {e}, using VQ codes")

    all_embeddings, all_labels = [], []
    for fi in range(0, len(dataset.graphs), stride):
        graph = dataset.graphs[fi].to(device)
        emb = model.get_embeddings(graph)
        if emb is None:
            print("  No GNN embeddings, skipping UMAP")
            return
        all_embeddings.append(emb)
        all_labels.extend(atom_labels)

    embeddings = np.concatenate(all_embeddings)
    n_frames = len(range(0, len(dataset.graphs), stride))
    print(f"  UMAP: {embeddings.shape[0]} points ({n_frames} frames)")

    coords = UMAP(n_components=2, random_state=42, n_neighbors=15,
                   min_dist=0.1, n_jobs=1).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = sorted(set(all_labels))
    for i, label in enumerate(unique_labels):
        mask = np.array([j for j, l in enumerate(all_labels) if l == label])
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], label=label, s=18, alpha=0.4,
                   edgecolors='white', linewidths=0.2, rasterized=True)

    ax.legend(fontsize=8, ncol=2, markerscale=2.0, framealpha=0.9,
              loc='best')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'GNN Atom Embeddings ({n_frames} frames)')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def write_structure_xyz(model, dataset, device, config, filepath):
    model.eval()
    graph0 = dataset.graphs[0].to(device)
    vq_result = model.get_atom_types(graph0)
    if vq_result is None:
        return

    from .labelling import compute_physical_labels

    symbols = list(dataset.frame_data[0][0])
    positions_bohr = np.asarray(dataset.frame_data[0][1])
    positions_ang = positions_bohr * BOHR_TO_ANGSTROM
    _, vq_type_names = vq_result

    inorganic = sorted(set(config.get('knn_edges', {}).get(
        'inorganic_elements', ['Cd', 'Se'])))
    cutoffs_bohr = getattr(dataset, 'cutoffs_bohr', None)
    if cutoffs_bohr is None:
        gap_cuts = getattr(dataset, 'gap_cuts_angstrom', {}) or {}
        cutoffs_bohr = {pair: cut * ANGSTROM_TO_BOHR for pair, cut in gap_cuts.items()}

    physical_labels, label_info = compute_physical_labels(
        symbols, positions_bohr, vq_type_names,
        inorganic, cutoffs_bohr,
        frame_data=dataset.frame_data, stride=10)

    # Colors: Cd=green (dark->bright by surface_score), Se=blue (dark->bright)
    cd_greens = [(0.0, 0.39, 0.0), (0.0, 0.7, 0.0), (0.0, 1.0, 0.0)]
    se_blues = [(0.0, 0.0, 0.6), (0.0, 0.45, 1.0), (0.3, 0.75, 1.0)]
    organic_colors = {
        'C': (0.5, 0.5, 0.5), 'H': (1.0, 1.0, 1.0), 'O': (1.0, 0.0, 0.0),
        'N': (0.0, 0.0, 1.0), 'S': (1.0, 1.0, 0.0), 'F': (0.0, 1.0, 0.0),
    }

    label_to_color = {}
    for elem, palette in [('Cd', cd_greens), ('Se', se_blues)]:
        elem_labels = sorted(
            [l for l in label_info if l.partition('_')[0] == elem and '_' in l],
            key=lambda l: label_info[l]['mean_surface_score'])
        for i, label in enumerate(elem_labels):
            label_to_color[label] = palette[min(i, len(palette) - 1)]

    n_atoms = len(physical_labels)
    with open(filepath, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f'Properties=species:S:1:pos:R:3:Color:R:3 '
                f'# VQ physical labels, frame 0\n')
        for i in range(n_atoms):
            x, y, z = positions_ang[i]
            label = physical_labels[i]
            color = label_to_color.get(label,
                        organic_colors.get(label, (0.5, 0.5, 0.5)))
            f.write(f"{label}  {x:.6f}  {y:.6f}  {z:.6f}  "
                    f"{color[0]:.3f}  {color[1]:.3f}  {color[2]:.3f}\n")
    print(f"Saved: {filepath}")


def generate_all_plots(model, dataset, device, config,
                       train_rmse_history=None, val_rmse_history=None,
                       output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    core_elements = config.get('core_elements', [])
    inorganic = config.get('knn_edges', {}).get('inorganic_elements', [])

    print(f"\nGenerating plots...")

    if train_rmse_history:
        plot_training(train_rmse_history, val_rmse_history or [],
                      os.path.join(output_dir, 'training_curves.png'))

    plot_force_parity(model, dataset, device,
                      os.path.join(output_dir, 'force_parity.png'),
                      core_elements=core_elements)

    plot_force_residuals(model, dataset, device,
                         os.path.join(output_dir, 'force_error_distribution.png'),
                         inorganic_elements=inorganic or None)

    plot_umap(model, dataset, device,
              os.path.join(output_dir, 'umap_embeddings.png'),
              stride=10, core_elements=core_elements or None,
              config=config)

    plot_cn_vs_label(model, dataset, device, config,
                     os.path.join(output_dir, 'cn_vs_label.png'),
                     stride=10)

    # Structure XYZ with VQ labels (for OVITO visualization)
    write_structure_xyz(model, dataset, device, config,
                        os.path.join(output_dir, 'structure_vq.xyz'))

    print(f"All plots saved to {output_dir}/")
