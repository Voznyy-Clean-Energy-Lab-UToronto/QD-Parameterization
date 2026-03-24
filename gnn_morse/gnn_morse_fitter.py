#!/usr/bin/env python3
import os
import sys
import time
import logging

import numpy as np
import yaml
import torch
from torch_geometric.loader import DataLoader

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    FORCE_AU_TO_EV_ANG,
)
from .data import DFTDataset
from .models import GNNMorseModel
from .lammps_export import export_lammps
from .plotting import generate_all_plots


def _setup_logging(log_path):
    fmt = logging.Formatter('%(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # Also tee stdout/stderr to the log file
    class _TeeStream:
        def __init__(self, stream, handler):
            self.stream = stream
            self.handler = handler
        def write(self, data):
            self.stream.write(data)
            if data.strip():
                self.handler.stream.write(data)
                self.handler.stream.flush()
        def flush(self):
            self.stream.flush()
    sys.stdout = _TeeStream(sys.stdout, fh)
    sys.stderr = _TeeStream(sys.stderr, fh)
    return fh


# ── Config ───────────────────────────────────────────────────────────

def load_config(filepath):
    with open(filepath) as f:
        raw = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(filepath))

    config = {}
    config.update(raw.get('training', {}))
    config['datasets'] = raw.get('datasets', [])
    if not config['datasets']:
        raise ValueError("Config must specify at least one dataset under 'datasets'")

    # Resolve relative xyz paths relative to config file location
    for ds in config['datasets']:
        if 'xyz' in ds and not os.path.isabs(ds['xyz']):
            ds['xyz'] = os.path.normpath(os.path.join(config_dir, ds['xyz']))

    # KNN config
    config['knn_edges'] = raw['knn_edges']

    # GNN config
    config['gnn'] = raw['gnn']

    # Fixed pairs
    fixed_raw = raw.get('fixed_pairs') or {}
    config['fixed_pairs'] = {}
    for name, values in fixed_raw.items():
        if isinstance(values, dict):
            config['fixed_pairs'][name] = {k: float(v) for k, v in values.items()}
        elif isinstance(values, (list, tuple)):
            config['fixed_pairs'][name] = {
                'D_e': float(values[0]), 'alpha': float(values[1]), 'r0': float(values[2])}

    # Initial guesses
    ig_raw = raw.get('initial_guesses') or {}
    config['initial_guesses'] = {}
    for name, values in ig_raw.items():
        if isinstance(values, dict):
            config['initial_guesses'][name] = {k: float(v) for k, v in values.items()}

    # Cast numeric fields (PyYAML reads 1e-3 as string)
    for key in ('learning_rate', 'minimum_learning_rate', 'convergence_threshold',
                'validation_split', 'weight_decay', 'energy_weight', 'max_hours',
                'box_size_angstrom'):
        if key in config:
            config[key] = float(config[key])

    # LAMMPS data file path (optional) — configs use 'original_data', fallback to 'lammps_data_file'
    od = raw.get('original_data') or raw.get('lammps_data_file')
    if od and not os.path.isabs(od):
        od = os.path.normpath(os.path.join(config_dir, od))
    config['original_data'] = od

    print(f"Loaded config: {filepath}")
    return config


# ── RDF-based r0 initialization ─────────────────────────────────────

def compute_rdf_peaks(pair_names, xyz_files, box_size):
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF
    except ImportError:
        print("MDAnalysis not available, skipping RDF peaks")
        return {}

    print("\nComputing equilibrium distances from RDF...")
    peaks = {}
    box_dims = [box_size, box_size, box_size, 90.0, 90.0, 90.0]

    for xyz_path in xyz_files:
        if not os.path.exists(xyz_path):
            continue
        try:
            u = mda.Universe(xyz_path, topology_format='XYZ')
            u.dimensions = box_dims

            for pn in pair_names:
                if pn in peaks:
                    continue
                e1, e2 = pn.split('-')
                sel1 = u.select_atoms(f"name {e1}")
                sel2 = u.select_atoms(f"name {e2}")
                if len(sel1) == 0 or len(sel2) == 0:
                    continue

                rdf = InterRDF(sel1, sel2, nbins=200, range=(0.5, 8.0))
                rdf.run()
                r = rdf.results.bins
                g = rdf.results.rdf

                # Find first peak
                from scipy.signal import argrelmax
                try:
                    peak_idx = argrelmax(g, order=5)[0]
                    if len(peak_idx) > 0:
                        r_peak = r[peak_idx[0]]
                        peaks[pn] = r_peak * ANGSTROM_TO_BOHR
                        print(f"  {pn}: r0 = {r_peak:.3f} A")
                except Exception:
                    pass
        except Exception as e:
            print(f"  RDF failed for {xyz_path}: {e}")

    return peaks


# ── Loss ─────────────────────────────────────────────────────────────

def compute_core_force_rmse(pred, ref, elem_idx, core_mask):
    sq_err = (pred - ref) ** 2
    if core_mask is not None:
        mask = core_mask[elem_idx].unsqueeze(1)
        return torch.sqrt((sq_err * mask).sum() / (mask.sum() * 3) + 1e-30)
    return torch.sqrt(sq_err.mean() + 1e-30)


# ── Phase timer ──────────────────────────────────────────────────────

class PhaseTimer:
    def __init__(self):
        self.phases = []
        self._start = time.time()
        self._phase_start = self._start

    def mark(self, name):
        now = time.time()
        self.phases.append((name, now - self._phase_start))
        self._phase_start = now

    def summary(self):
        total = time.time() - self._start
        print(f"\n{'='*60}")
        print("PHASE TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Phase':<30} {'Time':>10} {'%':>6}")
        print(f"  {'-'*48}")
        for name, dt in self.phases:
            print(f"  {name:<30} {dt:>9.1f}s {100*dt/total:>5.1f}%")
        print(f"  {'-'*48}")
        print(f"  {'TOTAL':<30} {total:>9.1f}s")
        print(f"{'='*60}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='gnn-morse',
        description='GNN-Morse: Train environment-aware Morse potentials and export to LAMMPS')
    parser.add_argument('config', help='Path to config YAML file')
    parser.add_argument('--output-dir', default=None,
                        help='Base output directory (default: directory containing config)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run LAMMPS MD and full analysis after training')
    parser.add_argument('--infante', action='store_true',
                        help='Include Infante reference MD in analysis (requires --analyze)')
    parser.add_argument('--lammps-bin', default=None,
                        help='Path to LAMMPS binary (default: $LAMMPS_BIN or lmp)')
    parser.add_argument('--nsteps', type=int, default=10000,
                        help='MD steps for analysis (default: 10000)')
    parser.add_argument('--temp', type=int, default=300,
                        help='MD temperature in K (default: 300)')
    args = parser.parse_args()

    timer = PhaseTimer()
    config = load_config(args.config)
    timer.mark("Config loading")

    output_base = args.output_dir
    if output_base is None:
        output_base = os.path.dirname(os.path.abspath(args.config))
    output_base = os.path.abspath(output_base)

    results_dir = os.path.join(output_base, 'results')
    lammps_dir = os.path.join(output_base, 'lammps')
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, 'training.log')
    log_handler = _setup_logging(log_path)
    print(f"Logging output to: {log_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GNN: {'enabled' if config['gnn']['enabled'] else 'disabled'}")
    lig_cut = config['knn_edges'].get('ligand_cutoff', 'N/A')
    print(f"KNN: k_cross={config['knn_edges']['k_cross']}, "
          f"max_gap_neighbors={config['knn_edges']['max_gap_neighbors']}, "
          f"ligand_cutoff={lig_cut} A")

    # ── Load data (CPU) ──
    dataset = DFTDataset(
        config['datasets'],
        knn_config=config['knn_edges'],
        first_n_frames=config.get('first_n_frames'),  # None = use all
    )

    timer.mark("Data loading")

    elements = dataset.elements
    pair_names = dataset.pair_names
    num_pairs = len(pair_names)
    num_elements = len(elements)
    pair_name_to_index = dataset.pair_name_to_index

    print(f"\nElements ({num_elements}): {elements}")
    print(f"Pairs ({num_pairs}): {pair_names}")

    # ── Initialize parameters ──
    init_D_e = np.full(num_pairs, 0.1 * EV_TO_HARTREE)
    init_alpha = np.full(num_pairs, 1.5 * BOHR_TO_ANGSTROM)
    init_r0 = np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)

    # Apply fixed_pairs
    frozen_pair_names = set()
    frozen_mask = torch.ones(num_pairs, dtype=torch.float64)

    for pn, vals in config.get('fixed_pairs', {}).items():
        if pn not in pair_name_to_index:
            continue
        idx = pair_name_to_index[pn]
        init_D_e[idx] = vals['D_e'] * EV_TO_HARTREE
        init_alpha[idx] = vals['alpha'] * BOHR_TO_ANGSTROM
        init_r0[idx] = vals['r0'] * ANGSTROM_TO_BOHR
        frozen_pair_names.add(pn)
        frozen_mask[idx] = 0.0

    # Apply initial guesses
    for pn, vals in config.get('initial_guesses', {}).items():
        if pn not in pair_name_to_index:
            continue
        idx = pair_name_to_index[pn]
        if 'D_e' in vals:
            init_D_e[idx] = vals['D_e'] * EV_TO_HARTREE
        if 'alpha' in vals:
            init_alpha[idx] = vals['alpha'] * BOHR_TO_ANGSTROM
        if 'r0' in vals:
            init_r0[idx] = vals['r0'] * ANGSTROM_TO_BOHR

    # Smart r0 from RDF peaks
    if config.get('use_smart_r0'):
        xyz_files = [ds['xyz'] for ds in config['datasets']]
        peaks = compute_rdf_peaks(pair_names, xyz_files,
                                  config['box_size_angstrom'])
        for pn, r0_bohr in peaks.items():
            if pn not in frozen_pair_names and pn in pair_name_to_index:
                init_r0[pair_name_to_index[pn]] = r0_bohr
    timer.mark("RDF peak computation")

    # ── Build graphs (CPU precompute) ──
    dataset.build_graphs()

    timer.mark("Graph building")

    if not dataset.graphs:
        print("ERROR: No graphs built!")
        sys.exit(1)

    # ── Train/val split (temporal — last val_frac of trajectory) ──
    n_total = len(dataset.graphs)
    n_val = max(1, int(n_total * config['validation_split']))
    n_train = n_total - n_val

    train_graphs = dataset.graphs[:n_train]
    val_graphs = dataset.graphs[n_train:]

    batch_size = config['batch_size']
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    print(f"\nTrain: {n_train} frames (0-{n_train-1}), "
          f"Val: {n_val} frames ({n_train}-{n_total-1}), Batch: {batch_size}")

    # ── Create model (move to device for training) ──
    model = GNNMorseModel(
        num_pairs=num_pairs,
        num_elements=num_elements,
        elements=elements,
        pair_names=pair_names,
        gnn_config=config['gnn'],
        init_D_e=init_D_e,
        init_alpha=init_alpha,
        init_r0=init_r0,
    ).to(device)

    frozen_mask = frozen_mask.to(device)

    # Core mask for force loss
    core_elements = config.get('core_elements', [])  # optional
    if core_elements:
        core_mask = torch.zeros(num_elements, dtype=torch.float64, device=device)
        for ce in core_elements:
            if ce in dataset.element_to_index:
                core_mask[dataset.element_to_index[ce]] = 1.0
        print(f"Core elements for loss: {core_elements}")
    else:
        core_mask = None

    # Disable energy offset if not using energy
    energy_weight = config['energy_weight']
    use_energy = energy_weight > 0
    if not use_energy:
        model.energy_offset.requires_grad_(False)
        print("Energy matching: disabled (force-only training)")
    else:
        print(f"Energy matching: weight = {energy_weight}")

    # Count parameters
    n_base = sum(p.numel() for p in [model.raw_D_e, model.raw_alpha, model.raw_r0])
    n_gnn = sum(p.numel() for n, p in model.named_parameters()
                if 'env_encoder' in n or 'param_predictor' in n)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_base} base Morse + {n_gnn} GNN = {n_total_params} total")

    # ── Optimizer ──
    lr = config['learning_rate']
    min_lr = config['minimum_learning_rate']
    wd = config['weight_decay']

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)

    # ── Training loop ──
    patience = config['convergence_patience']
    threshold = config['convergence_threshold']
    time_limit = config['max_hours'] * 3600

    best_rmse = float('inf')
    best_state = None
    epochs_no_improve = 0
    rmse_history, energy_history, val_history = [], [], []

    print(f"\n{'Epoch':>8} | {'Train RMSE':>10} | {'Val RMSE':>10} "
          f"| {'LR':>10} | {'Time':>8}")
    print("-" * 60)

    t0 = time.time()
    epoch = 0

    def _should_stop():
        nonlocal epochs_no_improve
        if time.time() - t0 > time_limit:
            print(f"\nTime limit reached ({time_limit / 3600:.1f} hours)")
            return True
        if epochs_no_improve >= patience:
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr <= min_lr * 1.01:
                print(f"\nConverged at minimum LR ({cur_lr:.0e})")
                return True
            # Drop LR by 10x and keep going
            new_lr = max(cur_lr / 10, min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            print(f"\n  Patience hit, LR: {cur_lr:.0e} -> {new_lr:.0e}")
            epochs_no_improve = 0
        return False

    while not _should_stop():
        epoch += 1
        elapsed = time.time() - t0

        # ── Train ──
        model.train()
        rmse_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        energy_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_batch = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            forces, energies = model(batch)
            force_rmse = compute_core_force_rmse(
                forces, batch.dft_forces, batch.element_indices, core_mask)

            loss = force_rmse
            if use_energy and batch.dft_energy is not None:
                energy_rmse = torch.sqrt(
                    torch.mean((energies - batch.dft_energy) ** 2) + 1e-30)
                loss = force_rmse + energy_weight * energy_rmse
                energy_sum += energy_rmse.detach()

            loss.backward()

            # Freeze fixed pairs
            if frozen_mask is not None:
                model.apply_frozen_mask(frozen_mask)

            optimizer.step()
            rmse_sum += force_rmse.detach()
            n_batch += 1

        train_rmse = (rmse_sum.item() / n_batch) * FORCE_AU_TO_EV_ANG
        train_energy = (energy_sum.item() / max(n_batch, 1)) * HARTREE_TO_EV

        # ── Validate ──
        model.eval()
        val_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_val_batch = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                forces, _ = model(batch)
                val_rmse = compute_core_force_rmse(
                    forces, batch.dft_forces, batch.element_indices, core_mask)
                val_sum += val_rmse
                n_val_batch += 1

        val_rmse_ev = (val_sum.item() / max(n_val_batch, 1)) * FORCE_AU_TO_EV_ANG

        rmse_history.append(train_rmse)
        energy_history.append(train_energy)
        val_history.append(val_rmse_ev)

        # Best model tracking
        if val_rmse_ev < best_rmse - threshold:
            best_rmse = val_rmse_ev
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        cur_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 1 or epoch <= 5 or epochs_no_improve == 0:
            print(f"{epoch:8d} | {train_rmse:10.6f} | {val_rmse_ev:10.6f} "
                  f"| {cur_lr:10.2e} | {elapsed:7.1f}s")

    timer.mark("Training")

    # ── Restore best ──
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val RMSE = {best_rmse:.6f} eV/A)")

    model.to(device)

    # ── Print final parameters ──
    params = model.get_base_params_display()
    print(f"\n{'='*60}")
    print("FINAL BASE MORSE PARAMETERS")
    if config['gnn']['enabled']:
        print("(Note: with GNN enabled, base params are shifted by per-edge corrections.")
        print(" Per-edge params in the LAMMPS export section show actual physical values.)")
    print('='*60)
    print(f"  {'Pair':>10} {'D_e(eV)':>12} {'alpha(1/A)':>12} {'r0(A)':>12}  Status")
    print(f"  {'-'*58}")
    for i, pn in enumerate(pair_names):
        status = "FROZEN" if pn in frozen_pair_names else "trained"
        print(f"  {pn:>10} {params['D_e'][i]:12.6f} {params['alpha'][i]:12.6f} "
              f"{params['r0'][i]:12.6f}  {status}")

    # ── Save checkpoint ──
    ckpt_path = os.path.join(results_dir, 'checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rmse': best_rmse,
        'config': config,
        'elements': elements,
        'pair_names': pair_names,
        'rmse_history': rmse_history,
        'val_history': val_history,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # ── Generate all diagnostic plots ──
    generate_all_plots(
        model, dataset, device, config,
        rmse_history=rmse_history,
        energy_history=energy_history,
        val_history=val_history,
        output_dir=os.path.join(results_dir, 'plots'),
    )

    timer.mark("Plotting")

    # ── LAMMPS export ──
    clustering_mode = config.get('lammps_export', {}).get('clustering', 'embedding')
    n_clusters = config.get('lammps_export', {}).get('n_clusters', None)
    export_lammps(model, dataset, config,
                  output_dir=lammps_dir,
                  clustering=clustering_mode, n_clusters=n_clusters,
                  original_data=config.get('original_data'),
                  write_lammps_in=False)
    timer.mark("LAMMPS export")

    print(f"\n{'='*60}")
    print(f"Done! Total time: {time.time() - t0:.1f}s")
    print(f"Best val RMSE: {best_rmse:.6f} eV/A")
    print(f"Results: {results_dir}")
    print(f"LAMMPS files: {lammps_dir}")
    print('='*60)

    timer.summary()

    log_handler.close()
    logging.getLogger().removeHandler(log_handler)
    sys.stdout = sys.stdout.stream if hasattr(sys.stdout, 'stream') else sys.stdout
    sys.stderr = sys.stderr.stream if hasattr(sys.stderr, 'stream') else sys.stderr
    print(f"Full log saved to: {log_path}")

    # ── Optional analysis ──
    if args.analyze:
        lammps_bin = args.lammps_bin or os.environ.get('LAMMPS_BIN', 'lmp')
        from .run_analysis import main_from_fitter
        main_from_fitter(
            run_dir=output_base,
            config_path=os.path.abspath(args.config),
            lammps_bin=lammps_bin,
            infante=args.infante,
            nsteps=args.nsteps,
            temp=args.temp,
        )


if __name__ == '__main__':
    main()
