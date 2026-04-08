#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import yaml
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from scipy.signal import argrelmax

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    FORCE_AU_TO_EV_ANG,
)
from .data import DFTDataset
from .models import GNNMorseModel
from .lammps_export import export_lammps
from .plotting import generate_all_plots


#saves the print statements to a .log file
class _TeeWriter:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
    def write(self, data):
        self.stream.write(data)
        if data:
            self.log_file.write(data)
            self.log_file.flush()
    def flush(self):
        self.stream.flush()


def _setup_logging(log_path):
    log_file = open(log_path, 'w')
    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)
    return log_file


#reading .yaml config file
def load_config(filepath):
    with open(filepath) as f:
        raw = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(filepath))
#filling in config file keys
    config = {}
    config.update(raw.get('training', {}))
    config['datasets'] = raw.get('datasets', [])
    if not config['datasets']:
        raise ValueError("Config must have at least one dataset")

    # Resolve relative xyz paths
    for ds in config['datasets']:
        if 'xyz' in ds and not os.path.isabs(ds['xyz']):
            ds['xyz'] = os.path.normpath(os.path.join(config_dir, ds['xyz']))

#Defines how many nearest neighbours to look at in determining first shell cutoff, and is paired with gap detection to build the graph. All in data.py. I do not like this#Solution at all. Gap detection and defining KNN edges is because I don't know a good way to avoid using CN, so I'm forcing myself to work with it. We need something
#Better than this.
    config['knn_edges'] = raw['knn_edges']
    config['gnn'] = raw['gnn']

    # Fixed pairs (frozen during training)
    fixed_raw = raw.get('fixed_pairs') or {}
    config['fixed_pairs'] = {}
    for name, vals in fixed_raw.items():
        if isinstance(vals, dict):
            config['fixed_pairs'][name] = {k: float(v) for k, v in vals.items()}
        elif isinstance(vals, (list, tuple)):
            config['fixed_pairs'][name] = {
                'D_e': float(vals[0]), 'alpha': float(vals[1]), 'r0': float(vals[2])}

    # Initial guesses (optional starting values, still trained)
    ig_raw = raw.get('initial_guesses') or {}
    config['initial_guesses'] = {}
    for name, vals in ig_raw.items():
        if isinstance(vals, dict):
            config['initial_guesses'][name] = {k: float(v) for k, v in vals.items()}

    # Ensure numeric types (PyYAML reads 1e-3 as string sometimes)
    for key in ('learning_rate', 'minimum_learning_rate', 'convergence_threshold',
                'validation_split', 'weight_decay', 'box_size_angstrom'):
        if key in config:
            config[key] = float(config[key])

    # Original LAMMPS .data file (for atom type remapping)
    original_data = raw.get('original_data') or raw.get('lammps_data_file')
    if original_data and not os.path.isabs(original_data):
        original_data = os.path.normpath(os.path.join(config_dir, original_data))
    config['original_data'] = original_data

    print(f"Loaded config: {filepath}")
    return config



def compute_rdf_peaks(pair_names, xyz_files, box_size):
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF
    except ImportError:
        print("MDAnalysis not available, skipping RDF peaks")
        return {}

    print("\nComputing equilibrium distances from RDF...")
    peaks = {}
    box = [box_size, box_size, box_size, 90.0, 90.0, 90.0]

    for xyz_path in xyz_files:
        if not os.path.exists(xyz_path):
            continue
        try:
            universe = mda.Universe(xyz_path, topology_format='XYZ')
            universe.dimensions = box

            for pair_name in pair_names:
                if pair_name in peaks:
                    continue
                elem_a, elem_b = pair_name.split('-')
                group_a = universe.select_atoms(f"name {elem_a}")
                group_b = universe.select_atoms(f"name {elem_b}")
                if len(group_a) == 0 or len(group_b) == 0:
                    continue

                rdf = InterRDF(group_a, group_b, nbins=200, range=(0.5, 8.0))
                rdf.run()

                try:
                    peak_indices = argrelmax(rdf.results.rdf, order=5)[0]
                    if len(peak_indices) > 0:
                        first_peak = peak_indices[0]
                        r_peak = rdf.results.bins[first_peak]
                        g_peak = rdf.results.rdf[first_peak]
                        peaks[pair_name] = (r_peak * ANGSTROM_TO_BOHR, g_peak)
                        print(f"  {pair_name}: r0 = {r_peak:.3f} A, g_peak = {g_peak:.1f}")
                except Exception:
                    pass
        except Exception as e:
            print(f"  RDF failed for {xyz_path}: {e}")

    return peaks


#Mask is so RMSE is only counted on some atoms (Cd and Se for now), so we aren't including ligands in F_RMSE, because they aren't being trained (just using CHARMM)
def force_rmse(predicted, reference, element_indices, core_mask):
    squared_error = (predicted - reference) ** 2
    if core_mask is not None:
        mask = core_mask[element_indices].unsqueeze(1)
        return torch.sqrt((squared_error * mask).sum() / (mask.sum() * 3) + 1e-30)
    return torch.sqrt(squared_error.mean() + 1e-30)



def main():
    import argparse
#arguements for when actually running the proram, including options for run_analysis.py 
    parser = argparse.ArgumentParser(
        prog='gnn-morse',
        description='Train GNN-Morse potentials and export to LAMMPS')
    parser.add_argument('config', help='Path to config YAML file')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: config file directory)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run LAMMPS MD analysis after training')
    parser.add_argument('--infante', action='store_true',
                        help='Include Infante reference (requires --analyze)')
    parser.add_argument('--lammps-bin', default=None,
                        help='LAMMPS binary path (default: $LAMMPS_BIN or lmp)')
    parser.add_argument('--nsteps', type=int, default=10000, help='MD steps')
    parser.add_argument('--temp', type=int, default=300, help='Temperature (K)')
    args = parser.parse_args()

    start_time = time.time()
    config = load_config(args.config)

    # Output directories
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))
    output_dir = os.path.abspath(output_dir)
    results_dir = os.path.join(output_dir, 'results')
    lammps_dir = os.path.join(output_dir, 'lammps')
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, 'training.log')
    log_file = _setup_logging(log_path)
    print(f"Logging output to: {log_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"KNN: max_gap_neighbors={config['knn_edges']['max_gap_neighbors']}, "
          f"ligand_cutoff={config['knn_edges'].get('ligand_cutoff', 'N/A')} A")

    # Load data
    dataset = DFTDataset(
        config['datasets'],
        knn_config=config['knn_edges'],
        first_n_frames=config.get('first_n_frames'),
    )

    elements = dataset.elements
    pair_names = dataset.pair_names
    num_pairs = len(pair_names)
    num_elements = len(elements)
    pair_name_to_index = dataset.pair_name_to_index

    print(f"\nElements ({num_elements}): {elements}")
    print(f"Pairs ({num_pairs}): {pair_names}")

    # Initialize Morse parameters
    init_D_e = np.full(num_pairs, 0.1 * EV_TO_HARTREE)
    init_alpha = np.full(num_pairs, 1.5 * BOHR_TO_ANGSTROM)
    init_r0 = np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)

    # Initial guesses (override defaults but still trained)
    for pair_name, vals in config.get('initial_guesses', {}).items():
        if pair_name not in pair_name_to_index:
            continue
        idx = pair_name_to_index[pair_name]
        if 'D_e' in vals:
            init_D_e[idx] = vals['D_e'] * EV_TO_HARTREE
        if 'alpha' in vals:
            init_alpha[idx] = vals['alpha'] * BOHR_TO_ANGSTROM
        if 'r0' in vals:
            init_r0[idx] = vals['r0'] * ANGSTROM_TO_BOHR

    #Initial guesses, if not defined in config, are made using the RDF. RDF computed the PMF, giving both initial guesses for r0 and potential well depth.
    if config.get('use_smart_r0'):
        xyz_files = [ds['xyz'] for ds in config['datasets']]
        peaks = compute_rdf_peaks(pair_names, xyz_files, config['box_size_angstrom'])
        kT_eV = 8.617e-5 * 300.0
        for pair_name, (r0_bohr, g_peak) in peaks.items():
            if pair_name in pair_name_to_index:
                idx = pair_name_to_index[pair_name]
                init_r0[idx] = r0_bohr
                if g_peak > 1.0:
                    init_D_e[idx] = kT_eV * np.log(g_peak) * EV_TO_HARTREE

    # Build graphs
    dataset.build_graphs()
    if not dataset.graphs:
        print("ERROR: No graphs built!")
        sys.exit(1)

    # Train/val split (temporal: last portion is validation)
    n_total = len(dataset.graphs)
    n_val = max(1, int(n_total * config['validation_split']))
    n_train = n_total - n_val

    batch_size = config['batch_size']
    train_loader = DataLoader(dataset.graphs[:n_train], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.graphs[n_train:], batch_size=batch_size, shuffle=False)

    print(f"\nTrain: {n_train} frames (0-{n_train-1}), "
          f"Val: {n_val} frames ({n_train}-{n_total-1}), Batch: {batch_size}")

    # VQ max codes: use gnn config directly, or fall back to atom_detail
    gnn_raw = config['gnn']
    if 'vq_max_codes' in gnn_raw:
        n_subtypes = gnn_raw['vq_max_codes']
        atom_detail = 'auto'

    # Create model
    gnn_config = gnn_raw.copy()
    gnn_config['core_elements'] = config.get('core_elements', elements)
    gnn_config['vq_max_codes'] = n_subtypes

    model = GNNMorseModel(
        num_pairs=num_pairs,
        num_elements=num_elements,
        elements=elements,
        pair_names=pair_names,
        gnn_config=gnn_config,
        init_D_e=init_D_e,
        init_k=2.0 * init_D_e * init_alpha ** 2, #instead of calculating alpha directly, calculate k, and use alpha = sqrt(k/2*De)
        init_r0=init_r0,
    ).to(device)

    # Core element mask (loss computed only on these atoms)
    core_elements = config.get('core_elements', [])
    if core_elements:
        core_mask = torch.zeros(num_elements, dtype=torch.float64, device=device)
        for elem in core_elements:
            if elem in dataset.element_to_index:
                core_mask[dataset.element_to_index[elem]] = 1.0
        print(f"Core elements for loss: {core_elements}")
    else:
        core_mask = None

    # Parameter count
    n_base = sum(p.numel() for p in [model.raw_D_e, model.raw_k, model.raw_r0])
    n_gnn = sum(p.numel() for name, p in model.named_parameters()
                if 'env_encoder' in name or 'param_predictor' in name)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_base} base Morse + {n_gnn} GNN = {n_total_params} total")

    # Initialize codebook from first batch. Codebook is the quantized atom embeddings. 
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(train_loader)).to(device)
        h_init = model.env_encoder(
            first_batch.element_indices, first_batch.edge_index, first_batch.distances)
        model.vq.initialize_codebook(h_init, first_batch.element_indices)
    model.vq_active = True

    active_codes = model.vq.get_active_codes()
    for elem, n in active_codes.items():
        if model.vq.n_codes_per_elem[elem] > 1:
            print(f"  {elem}: {model.vq.n_codes_per_elem[elem]} codes initialized")

    # Optimizer: GNN weights get weight_decay, Morse base params get none
    weight_decay = config.get('weight_decay', 1e-4)
    lr = config['learning_rate']

    base_params = [model.raw_D_e, model.raw_k, model.raw_r0]
    base_param_ids = {id(p) for p in base_params}
    gnn_params = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in base_param_ids]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'weight_decay': 0.0},
        {'params': gnn_params, 'weight_decay': weight_decay},
    ], lr=lr)

    # Cosine LR schedule
    max_epochs = config.get('max_epochs', 300)
    min_lr = config.get('minimum_learning_rate', lr * 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=min_lr)

    print(f"Optimizer: Adam, lr={lr}, weight_decay={weight_decay} (GNN only)")
    print(f"Schedule: cosine annealing over {max_epochs} epochs, min_lr={min_lr:.2e}")

    # Training loop
    patience = config['convergence_patience']
    threshold = config['convergence_threshold']

    best_val_rmse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    train_rmse_history = []
    val_rmse_history = []

    print(f"\n{'Epoch':>8} | {'Train RMSE':>10} | {'Val RMSE':>10} "
          f"| {'LR':>10} | {'Time':>8}")
    print("-" * 60)

    train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        elapsed = time.time() - train_start

        # Check patience -> stop training
        if epochs_without_improvement >= patience:
            print(f"\nConverged (no improvement for {patience} epochs)")
            break

        # Train one epoch
        model.train()
        batch_rmse_total = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            forces_pred, vq_loss = model(batch)
            rmse = force_rmse(forces_pred, batch.dft_forces, batch.element_indices, core_mask)

            loss = rmse
            if model.vq_active:
                loss = loss + vq_loss

            loss.backward()
            optimizer.step()

            batch_rmse_total += rmse.item()
            num_batches += 1

        train_rmse_ev = (batch_rmse_total / num_batches) * FORCE_AU_TO_EV_ANG

        # Validate
        model.eval()
        val_rmse_total = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                forces_pred, _ = model(batch)
                rmse = force_rmse(forces_pred, batch.dft_forces, batch.element_indices, core_mask)
                val_rmse_total += rmse.item()
                num_val_batches += 1

        val_rmse_ev = (val_rmse_total / num_val_batches) * FORCE_AU_TO_EV_ANG

        train_rmse_history.append(train_rmse_ev)
        val_rmse_history.append(val_rmse_ev)

        # Track best model
        if val_rmse_ev < best_val_rmse - threshold:
            best_val_rmse = val_rmse_ev
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        if epoch <= 5 or epoch % 10 == 1 or epochs_without_improvement == 0:
            print(f"{epoch:8d} | {train_rmse_ev:10.6f} | {val_rmse_ev:10.6f} "
                  f"| {current_lr:10.2e} | {elapsed:7.1f}s")

        # VQ monitoring every 50 epochs
        if model.vq_active and epoch % 50 == 0:
            active = model.vq.get_active_codes()
            parts = []
            for elem, n_active in active.items():
                if model.vq.n_codes_per_elem[elem] > 1:
                    parts.append(f"{elem}:{n_active}/{model.vq.n_codes_per_elem[elem]}")
            if parts:
                print(f"  VQ [{epoch}]: {', '.join(parts)}")

        scheduler.step()

    # Restore best model
    if best_model_state is not None:
        vq_was_active = model.vq_active
        model.load_state_dict(best_model_state)
        model.vq_active = vq_was_active
        print(f"\nRestored best model (val RMSE = {best_val_rmse:.6f} eV/A)")

    model.to(device)

    # VQ summary
    if model.vq_active:
        active_codes = model.vq.get_active_codes()
        print(f"\nVQ CODEBOOK SUMMARY (after training)")
        for elem, n_active in active_codes.items():
            if model.vq.n_codes_per_elem[elem] > 1:
                print(f"  {elem}: {n_active} active codes "
                      f"(of {model.vq.n_codes_per_elem[elem]} max)")

    # Print final Morse parameters
    params = model.get_base_params_display()
    print(f"\n{'='*60}")
    print("FINAL BASE MORSE PARAMETERS")
    print("(Note: base params are shifted by per-edge GNN corrections.")
    print(" Actual values are in the LAMMPS export section.)")
    print('='*60)
    print(f"  {'Pair':>10} {'D_e(eV)':>12} {'alpha(1/A)':>12} {'r0(A)':>12} {'k(eV/A^2)':>12}")
    print(f"  {'-'*62}")
    for i, pair_name in enumerate(pair_names):
        print(f"  {pair_name:>10} {params['D_e'][i]:12.6f} {params['alpha'][i]:12.6f} "
              f"{params['r0'][i]:12.6f} {params['k'][i]:12.6f}")

    # Save checkpoint
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rmse': best_val_rmse,
        'config': config,
        'elements': elements,
        'pair_names': pair_names,
        'train_rmse_history': train_rmse_history,
        'val_rmse_history': val_rmse_history,
        'vq_active': model.vq_active,
    }, checkpoint_path)
    print(f"Saved: {checkpoint_path}")

    # Save per-edge reference data (label-invariant, for consistency comparison)
    model.eval()
    with torch.no_grad():
        ref_batch = dataset.graphs[0].to(device)
        # Unsqueeze to add batch dim for single graph
        ref_batch = Batch.from_data_list([ref_batch])
        ref_batch = ref_batch.to(device)

        ref_forces, _ = model(ref_batch)
        ref_edge_params = model.get_per_edge_params(ref_batch)

        ref_path = os.path.join(results_dir, 'reference_frame.npz')
        np.savez(ref_path,
                 # Per-edge params (label-invariant)
                 edge_D_e=ref_edge_params['D_e'],
                 edge_k=ref_edge_params['k'],
                 edge_alpha=ref_edge_params['alpha'],
                 edge_r0=ref_edge_params['r0'],
                 # Edge metadata
                 pair_indices=ref_batch.pair_indices.cpu().numpy(),
                 pair_names=np.array(pair_names),
                 edge_distances=ref_batch.distances.cpu().numpy() * BOHR_TO_ANGSTROM,
                 # Forces
                 predicted_forces=ref_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG,
                 dft_forces=ref_batch.dft_forces.cpu().numpy() * FORCE_AU_TO_EV_ANG,
                 # Atom metadata
                 element_indices=ref_batch.element_indices.cpu().numpy(),
                 elements=np.array(elements),
                 )
        print(f"Saved reference frame data: {ref_path}")

    # Diagnostic plots
    generate_all_plots(
        model, dataset, device, config,
        train_rmse_history=train_rmse_history,
        val_rmse_history=val_rmse_history,
        output_dir=os.path.join(results_dir, 'plots'),
    )

    # LAMMPS export
    fallback_De = config.get('fallback_De', 0.001)

    export_lammps(model, dataset, config,
                  output_dir=lammps_dir,
                  original_data=config.get('original_data'),
                  atom_detail=atom_detail,
                  fallback_De=fallback_De)

    # Done
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Total time: {total_time:.1f}s")
    print(f"Best val RMSE: {best_val_rmse:.6f} eV/A")
    print(f"Results: {results_dir}")
    print(f"LAMMPS files: {lammps_dir}")
    print('='*60)

    # Cleanup logging
    log_file.close()
    sys.stdout = sys.stdout.stream
    sys.stderr = sys.stderr.stream
    print(f"Full log saved to: {log_path}")

    # Optional post-training analysis
    if args.analyze:
        lammps_bin = args.lammps_bin or os.environ.get('LAMMPS_BIN', 'lmp')
        from .run_analysis import main_from_fitter
        main_from_fitter(
            run_dir=output_dir,
            config_path=os.path.abspath(args.config),
            lammps_bin=lammps_bin,
            infante=args.infante,
            nsteps=args.nsteps,
            temp=args.temp,
            atom_detail=atom_detail,
        )


if __name__ == '__main__':
    main()
