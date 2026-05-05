import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch_geometric.loader import DataLoader

from .utils import BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG, ORGANIC_ELEMENTS, base_element


def _collect_forces(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset.graphs, batch_size=32, shuffle=False)
    pred_all, ref_all, elem_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            forces = model(batch)
            pred_all.append(forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            ref_all.append(batch.dft_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG)
            elem_all.append(batch.element_indices.cpu().numpy())
    return np.concatenate(pred_all), np.concatenate(ref_all), np.concatenate(elem_all)


def plot_training(train_history, val_history, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_history) + 1)
    ax.plot(epochs, train_history, 'b-', label='Train', alpha=0.8)
    if val_history:
        ax.plot(epochs, val_history, 'r-', label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (eV/A)')
    ax.set_title(f'Training Curves (best val = {min(val_history):.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_force_parity(model, dataset, device, filepath, core_elements=None):
    pred, ref, elem_idx = _collect_forces(model, dataset, device)
    elements = dataset.elements

    if core_elements:
        core_mask = np.array([elements[ei] in set(core_elements) for ei in elem_idx])
    else:
        core_mask = np.ones(len(elem_idx), dtype=bool)

    core_3d = np.repeat(core_mask, 3)
    pred_core = pred.ravel()[core_3d]
    ref_core = ref.ravel()[core_3d]
    rmse = np.sqrt(np.mean((pred_core - ref_core) ** 2))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(ref_core, pred_core, gridsize=80, mincnt=1, cmap='viridis', norm=LogNorm())
    lim = np.percentile(np.abs(np.concatenate([ref_core, pred_core])), 99.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('DFT Force (eV/A)')
    ax.set_ylabel('Predicted Force (eV/A)')
    ax.set_title(f'Force Parity | RMSE = {rmse:.4f} eV/A')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.8, label='Count')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_force_residuals(model, dataset, device, filepath, inorganic_elements=None):
    pred, ref, elem_idx = _collect_forces(model, dataset, device)
    elements = dataset.elements
    sq_err = np.sum((pred - ref) ** 2, axis=1)

    show = set(inorganic_elements) if inorganic_elements else set(elements)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(sq_err, 99), 80)
    colors = plt.cm.Set1(np.linspace(0, 1, len(elements)))

    for ei, elem in enumerate(elements):
        if elem not in show:
            continue
        mask = elem_idx == ei
        if not mask.any():
            continue
        vals = sq_err[mask]
        ax.hist(vals, bins=bins, alpha=0.6, color=colors[ei], edgecolor='none',
                label=f'{elem} (med={np.median(vals):.4f})')

    ax.set_xlabel('|F_DFT - F_pred|^2 (eV/A)^2')
    ax.set_ylabel('Count')
    ax.set_title('Force Error Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_temporal_correlation(model, dataset, device, filepath, n_pairs=200):
    model.eval()
    elements = dataset.elements
    stride = max(1, (len(dataset.graphs) - 1) // n_pairs)

    all_dF_pred, all_dF_dft = [], []
    with torch.no_grad():
        for i in range(0, len(dataset.graphs) - 1, stride):
            g0 = dataset.graphs[i].to(device)
            g1 = dataset.graphs[i + 1].to(device)
            all_dF_pred.append((model(g1) - model(g0)).cpu().numpy())
            all_dF_dft.append((g1.dft_forces - g0.dft_forces).cpu().numpy())

    dF_pred = np.concatenate(all_dF_pred)
    dF_dft = np.concatenate(all_dF_dft)
    n_frames = len(all_dF_pred)
    elem_idx = dataset.graphs[0].element_indices.numpy()

    fig, axes = plt.subplots(1, len(elements), figsize=(5 * len(elements), 5), squeeze=False)
    for ax, (ei, elem) in zip(axes[0], enumerate(elements)):
        mask = np.tile(elem_idx == ei, n_frames)
        if not mask.any():
            continue
        p = dF_pred[mask].ravel()
        d = dF_dft[mask].ravel()
        corr = np.corrcoef(p, d)[0, 1]
        ax.hexbin(d, p, gridsize=60, mincnt=1, cmap='viridis', norm=LogNorm())
        lim = np.percentile(np.abs(np.concatenate([p, d])), 99) * 1.2
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('dF_DFT (eV/A)')
        ax.set_ylabel('dF_pred (eV/A)')
        ax.set_title(f'{elem}: corr={corr:.3f}')
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_morse_potentials(model, cpair_names, cpair_to_base, cutoffs_ang,
                          subtype_cutoffs_ang, output_file):
    params = model.get_type_pair_params()
    base_pairs = ['Cd-Se', 'Cd-Cd', 'Se-Se', 'Cd-O', 'O-Se']

    active = []
    for bp in base_pairs:
        subtypes = [(cp, params[cp]) for cp in cpair_names
                    if cpair_to_base.get(cp) == bp and params[cp]['D_e'] > 0.005]
        if subtypes:
            active.append((bp, subtypes))
    if not active:
        return

    r = np.linspace(0.5, 10.0, 500)
    fig, axes = plt.subplots(len(active), 1, figsize=(12, 3.5 * len(active)))
    axes = np.atleast_1d(axes)
    cmap = plt.cm.tab10

    for ax, (bp, subtypes) in zip(axes, active):
        subtypes.sort(key=lambda x: -x[1]['D_e'])
        max_D = max(p['D_e'] for _, p in subtypes)

        for idx, (cp_name, p) in enumerate(subtypes):
            D, a, r0 = p['D_e'], p['alpha'], p['r0']
            rc = subtype_cutoffs_ang.get(cp_name, cutoffs_ang.get(bp, 7.5))

            exp1 = np.exp(-a * (r - r0))
            v = D * (1 - exp1)**2 - D
            v_at_cut = D * (1 - np.exp(-a * (rc - r0)))**2 - D
            v = v - v_at_cut
            v[r > rc] = 0.0
            v = np.clip(v, -max_D * 1.5, max_D * 1.5)

            color = cmap(idx % 10)
            ax.plot(r, v, color=color, lw=1.5,
                    label=f'{cp_name} D={D:.3f} r0={r0:.2f}')
            ax.axvline(rc, color=color, ls='--', lw=0.8, alpha=0.5)

        ax.axhline(0, color='gray', lw=1, alpha=0.4)
        ax.set_xlim(1.5, 7.0)
        ax.set_ylim(-max_D * 1.5, max_D * 1.5)
        ax.set_ylabel('V (eV)')
        ax.set_title(bp)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('r (Angstrom)')
    fig.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def write_structure_xyz(positions_ang, symbols, atom_type_names, inorganic_elements, filepath):
    """Write colored XYZ for visualizing subtypes in OVITO."""
    n = len(symbols)
    inorg = set(inorganic_elements)
    center = positions_ang.mean(axis=0)
    centered = positions_ang - center
    box = (np.max(np.abs(centered)) + 5.0) * 2

    palette_g = [(0.0, 0.4, 0.0), (0.0, 0.7, 0.0), (0.4, 1.0, 0.2)]
    palette_b = [(0.0, 0.0, 0.6), (0.0, 0.6, 0.9), (0.4, 0.8, 1.0)]
    unique_types = sorted(set(atom_type_names))
    type_colors = {}
    for elem in sorted(inorg):
        pal = palette_g if elem == 'Cd' else palette_b
        et = sorted(t for t in unique_types if base_element(t) == elem)
        for i, t in enumerate(et):
            type_colors[t] = pal[min(i, len(pal) - 1)]

    with open(filepath, 'w') as f:
        f.write(f"{n}\n")
        f.write(f'Lattice="{box:.1f} 0 0 0 {box:.1f} 0 0 0 {box:.1f}" '
                f'Properties=species:S:1:pos:R:3:Subtype:S:1:Color:R:3:Radius:R:1\n')
        for i in range(n):
            c = type_colors.get(atom_type_names[i], (0.6, 0.6, 0.6))
            r = 1.2 if symbols[i] in inorg else 0.6
            f.write(f"{symbols[i]} {centered[i,0]:.6f} {centered[i,1]:.6f} {centered[i,2]:.6f} "
                    f"{atom_type_names[i]} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {r:.1f}\n")
    print(f"  Saved: {filepath}")


def generate_all_plots(model, dataset, device, config, cpair_names=None, cpair_to_base=None,
                       train_rmse_history=None, val_rmse_history=None,
                       atom_type_names=None, output_dir='.'):
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

    if len(dataset.graphs) > 1:
        plot_temporal_correlation(model, dataset, device,
                                 os.path.join(output_dir, 'temporal_correlation.png'))

    if cpair_names and cpair_to_base:
        sub_cuts = getattr(dataset, 'subtype_cutoffs_angstrom', {})
        elem_cuts = getattr(dataset, 'gap_cuts_angstrom', {})
        plot_morse_potentials(model, cpair_names, cpair_to_base,
                              elem_cuts, sub_cuts,
                              os.path.join(output_dir, 'morse_potentials.png'))

    if atom_type_names and inorganic:
        symbols = dataset.frame_data[0][0]
        pos_ang = np.asarray(dataset.frame_data[0][1]) * BOHR_TO_ANGSTROM
        write_structure_xyz(pos_ang, symbols, atom_type_names, inorganic,
                           os.path.join(output_dir, 'structure_subtypes.xyz'))

    print(f"  All plots saved to {output_dir}/")
