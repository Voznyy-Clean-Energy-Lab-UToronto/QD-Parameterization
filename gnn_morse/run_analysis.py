#!/usr/bin/env python3
"""
Full analysis pipeline: run LAMMPS MD, analyze trajectory, force rerun.

Usage (from repo root):
    python -m gnn_morse_local.run_analysis gnn_morse_local/runs/Cd68Se55_300K/

Options:
    --skip-md       Skip MD, just analyze existing trajectories
    --skip-rerun    Skip force rerun
    --no-infante    Skip Infante reference MD
    --lammps-bin    Path to LAMMPS binary (default: lmp)

Outputs (all in run_dir/analysis/):
    rdf.png              - RDF comparison (DFT vs GNN-Morse vs Infante)
    structural.png       - Structural phase space density
    energy_stability.png - PE and temperature over time
    potentials.png       - Cluster sub-type curves + Infante reference
    force_rerun.png      - Force parity + per-frame RMSE
"""

import os
import sys
import io
import subprocess
import argparse
import re

import numpy as np
import yaml
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import base_element, FROZEN_LJ


# ============================================================
# CONSTANTS
# ============================================================

RDF_R_MAX = 12.0
RDF_BIN_WIDTH = 0.05
BOND_CUTOFF = 3.5
SKIP_FRAMES = 10

ATOM_MAP = {'Cd': 1, 'Se': 2, 'C': 3, 'H': 4, 'O': 5}
ATOM_MAP_INV = {v: k for k, v in ATOM_MAP.items()}

COULOMB_CONST = 14.3996

INFANTE_REF = {
    'q_1': 0.9768, 'q_2': -0.9768, 'q_3': 0.4524, 'q_4': 0.0, 'q_5': -0.4704,
    'pair_1_1': ('lj', 0.0032139609, 1.234),
    'pair_1_2': ('lj', 0.0157796047, 2.94),
    'pair_1_3': ('lj', 0.0020887769, 1.9894040478),
    'pair_1_4': ('lj', 0.0005437367, 1.8726951615),
    'pair_1_5': ('lj', 0.0190080755, 2.471),
    'pair_2_2': ('lj', 0.0044213986, 4.852),
    'pair_2_3': ('lj', 0.0024499183, 3.9448074845),
    'pair_2_4': ('lj', 0.0006377467, 3.7133843663),
    'pair_2_5': ('lj', 0.0167227534, 3.526),
    'pair_3_3': ('lj', 0.0013575115, 3.2072353853),
    'pair_3_4': ('lj', 0.0003533785, 3.0190821189),
    'pair_3_5': ('lj', 0.0014280135, 3.2072353853),
    'pair_4_4': ('lj', 9.19892e-05, 2.8419669109),
    'pair_4_5': ('lj', 0.0003717311, 3.0190821189),
    'pair_5_5': ('lj', 0.001502177, 3.2072353853),
}

# LAMMPS pair_coeff block for Infante LJ+Coulomb (5 atom types: Cd=1 Se=2 C=3 H=4 O=5)
INFANTE_LAMMPS = """\
# Infante LJ+Coulomb reference
set              type 1 charge  0.9768
set              type 2 charge -0.9768
set              type 3 charge  0.4524
set              type 4 charge  0.0
set              type 5 charge -0.4704

pair_style       lj/cut/coul/cut 60.0

pair_coeff       1 1  0.0032139609  1.234
pair_coeff       1 2  0.0157796047  2.94
pair_coeff       1 3  0.0020887769  1.9894040478
pair_coeff       1 4  0.0005437367  1.8726951615
pair_coeff       1 5  0.0190080755  2.471
pair_coeff       2 2  0.0044213986  4.852
pair_coeff       2 3  0.0024499183  3.9448074845
pair_coeff       2 4  0.0006377467  3.7133843663
pair_coeff       2 5  0.0167227534  3.526
pair_coeff       3 3  0.0013575115  3.2072353853
pair_coeff       3 4  0.0003533785  3.0190821189
pair_coeff       3 5  0.0014280135  3.2072353853
pair_coeff       4 4  9.19892e-05   2.8419669109
pair_coeff       4 5  0.0003717311  3.0190821189
pair_coeff       5 5  0.001502177   3.2072353853
"""


# ============================================================
# TRAJECTORY PARSING
# ============================================================

def parse_xyz_frames(filepath, skip_frames=0):
    if not os.path.exists(filepath):
        print(f"  [Error] Not found: {filepath}")
        return []
    with open(filepath) as f:
        content = f.read()
    if 'ITEM: TIMESTEP' in content:
        return _parse_lammps_dump(content, skip_frames)
    return _parse_standard_xyz(content, skip_frames)


def _parse_lammps_dump(content, skip_frames=0):
    frames = []
    sections = content.split('ITEM: TIMESTEP')
    for frame_idx, section in enumerate(sections[1:], start=1):
        lines = section.strip().split('\n')
        try:
            natoms_idx = next(i for i, l in enumerate(lines) if 'NUMBER OF ATOMS' in l)
            natoms = int(lines[natoms_idx + 1].strip())
            atoms_idx = next(i for i, l in enumerate(lines) if 'ITEM: ATOMS' in l)
        except (StopIteration, ValueError, IndexError):
            continue
        header = lines[atoms_idx].replace('ITEM: ATOMS', '').strip().split()
        col = {name: i for i, name in enumerate(header)}
        elem_col = col.get('element', 0)
        x_col = col.get('xu', col.get('x', 1))
        y_col, z_col = x_col + 1, x_col + 2
        atom_lines = lines[atoms_idx + 1: atoms_idx + 1 + natoms]
        if len(atom_lines) < natoms:
            continue
        atoms = []
        coords = np.empty((natoms, 3))
        for j, line in enumerate(atom_lines):
            parts = line.split()
            atoms.append(parts[elem_col])
            coords[j] = [float(parts[x_col]), float(parts[y_col]), float(parts[z_col])]
        if frame_idx > skip_frames:
            frames.append({'atoms': np.array(atoms), 'coords': coords})
    return frames


def _parse_standard_xyz(content, skip_frames=0):
    frames = []
    lines = content.strip().split('\n')
    i = 0
    frame_idx = 0
    while i < len(lines):
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        i += 2
        atom_lines = lines[i:i + natoms]
        i += natoms
        frame_idx += 1
        if frame_idx <= skip_frames or len(atom_lines) < natoms:
            continue
        atoms = []
        coords = np.empty((natoms, 3))
        for j, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(parts[0])
            coords[j] = [float(parts[1]), float(parts[2]), float(parts[3])]
        if atoms:
            frames.append({'atoms': np.array(atoms), 'coords': coords})
    return frames


# ============================================================
# RDF
# ============================================================

def compute_rdf(frames, atom1, atom2, box=None, r_max=RDF_R_MAX, bin_width=RDF_BIN_WIDTH):
    n_bins = int(r_max / bin_width)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)
    n_frames = 0
    same_species = (atom1 == atom2)
    for frame in frames:
        atoms = frame['atoms']
        coords = frame['coords']
        idx1 = np.where(atoms == atom1)[0]
        idx2 = np.where(atoms == atom2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        n_frames += 1
        dists = cdist(coords[idx1], coords[idx2]).ravel()
        if same_species:
            dists = dists[dists > 0.01]
        hist += np.histogram(dists, bins=r_edges)[0]
    if n_frames == 0:
        return r_bins, np.zeros_like(r_bins)
    if box is not None:
        vol = box[0] * box[1] * box[2]
        avg_n1 = np.mean([np.sum(f['atoms'] == atom1) for f in frames])
        avg_n2 = np.mean([np.sum(f['atoms'] == atom2) for f in frames])
        rho = avg_n2 / vol
        shell_vol = (4 / 3) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
        ideal = avg_n1 * rho * shell_vol * n_frames
        hist = np.where(ideal > 0, hist / ideal, 0)
    return r_bins, hist


# ============================================================
# STRUCTURAL ANALYSIS
# ============================================================

def analyze_structure(frames, center='Cd', neighbor='Se', cutoff=BOND_CUTOFF):
    paired_dists = []
    paired_angles = []
    for frame in frames:
        atoms = frame['atoms']
        coords = frame['coords']
        center_idx = np.where(atoms == center)[0]
        neighbor_idx = np.where(atoms == neighbor)[0]
        if len(center_idx) == 0 or len(neighbor_idx) == 0:
            continue
        center_pos = coords[center_idx]
        neighbor_pos = coords[neighbor_idx]
        all_dists = cdist(center_pos, neighbor_pos)
        for ci in range(len(center_pos)):
            dists_row = all_dists[ci]
            bond_mask = dists_row < cutoff
            bonded_dists = dists_row[bond_mask]
            if len(bonded_dists) < 2:
                continue
            bonded_vecs = neighbor_pos[bond_mask] - center_pos[ci]
            unit_vecs = bonded_vecs / bonded_dists[:, np.newaxis]
            cos_matrix = np.clip(unit_vecs @ unit_vecs.T, -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_matrix))
            tri_i, tri_j = np.triu_indices(len(unit_vecs), k=1)
            pair_angles = angles[tri_i, tri_j]
            paired_dists.append(bonded_dists[tri_i])
            paired_dists.append(bonded_dists[tri_j])
            paired_angles.append(pair_angles)
            paired_angles.append(pair_angles)
    if paired_dists:
        return np.concatenate(paired_dists), np.concatenate(paired_angles)
    return np.array([]), np.array([])


# ============================================================
# LOG PARSING
# ============================================================

def parse_lammps_log(filepath):
    data = {'step': [], 'temp': [], 'pe': [], 'ke': [], 'etotal': [], 'press': []}
    in_thermo = False
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('Step '):
                in_thermo = True
                continue
            if in_thermo:
                if stripped.startswith('Loop time') or not stripped:
                    in_thermo = False
                    continue
                parts = stripped.split()
                if len(parts) >= 6:
                    try:
                        data['step'].append(int(parts[0]))
                        data['temp'].append(float(parts[1]))
                        data['pe'].append(float(parts[2]))
                        data['ke'].append(float(parts[3]))
                        data['etotal'].append(float(parts[4]))
                        data['press'].append(float(parts[5]))
                    except (ValueError, IndexError):
                        in_thermo = False
    return {k: np.array(v) for k, v in data.items()} if data['step'] else None


# ============================================================
# POTENTIALS PARSING (cluster sub-types from potentials.txt)
# ============================================================

def parse_cluster_potentials(filepath):
    """Parse potentials.txt, extracting per-cluster-pair Morse params.

    Returns:
        cluster_params: {base_pair: [(cluster_name, D_e, alpha, r0, is_fallback), ...]}
            e.g. {'Cd-Se': [('Cd_0-Se_2', 0.68, 1.88, 2.57, False), ...]}
        lj_params: {base_pair: (eps, sigma)}  for frozen organic LJ pairs
    """
    if not os.path.exists(filepath):
        return {}, {}

    cluster_params = {}
    lj_params = {}

    # Detect pair_style from file to know format
    is_hybrid = False
    with open(filepath) as f:
        for line in f:
            if line.strip().startswith('pair_style') and 'hybrid' in line:
                is_hybrid = True
                break

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or not stripped.startswith('pair_coeff'):
                continue

            # Extract comment for cluster name
            comment = ''
            if '#' in stripped:
                comment = stripped.split('#', 1)[1].strip()
                stripped = stripped.split('#')[0].strip()

            parts = stripped.split()
            if len(parts) < 5:
                continue

            try:
                if is_hybrid:
                    # hybrid format: pair_coeff T1 T2 morse D_e alpha r0 cut
                    style = parts[3]
                    if style == 'morse' and len(parts) >= 8:
                        D_e, alpha, r0 = float(parts[4]), float(parts[5]), float(parts[6])
                        is_fallback = '[fallback]' in comment.lower()
                        cluster_name = comment.split()[0] if comment else ''
                        if '-' in cluster_name:
                            ce1, ce2 = cluster_name.split('-')
                            bp = '-'.join(sorted([base_element(ce1), base_element(ce2)]))
                            cluster_params.setdefault(bp, []).append(
                                (cluster_name, D_e, alpha, r0, is_fallback))
                    elif style == 'lj/cut' and len(parts) >= 6:
                        eps, sigma = float(parts[4]), float(parts[5])
                        if comment:
                            pair_name = comment.split('(')[0].strip()
                            if '-' in pair_name:
                                lj_params[pair_name] = (eps, sigma)
                else:
                    # direct morse format: pair_coeff T1 T2 D_e alpha r0 cut
                    if len(parts) >= 7:
                        D_e, alpha, r0 = float(parts[3]), float(parts[4]), float(parts[5])
                        is_fallback = '[fallback]' in comment.lower()
                        cluster_name = comment.split()[0] if comment else ''
                        if '-' in cluster_name:
                            ce1, ce2 = cluster_name.split('-')
                            bp = '-'.join(sorted([base_element(ce1), base_element(ce2)]))
                            cluster_params.setdefault(bp, []).append(
                                (cluster_name, D_e, alpha, r0, is_fallback))
            except (ValueError, IndexError):
                continue

    return cluster_params, lj_params


# ============================================================
# INFANTE MD
# ============================================================

def run_infante_md(original_data, results_dir, temp=300, nsteps=10000, lammps_bin='lmp'):
    """Run Infante LJ+Coulomb MD using the original .data file."""
    if not original_data or not os.path.exists(original_data):
        print(f"  [Skip] Original .data not found: {original_data}")
        return None, None

    infante_dir = os.path.join(results_dir, 'infante')
    os.makedirs(infante_dir, exist_ok=True)

    infante_xyz = os.path.join(infante_dir, 'infante.xyz')
    infante_log = os.path.join(infante_dir, 'infante.log')
    infante_in = os.path.join(infante_dir, 'infante.in')

    with open(infante_in, 'w') as f:
        f.write(f"# Infante LJ+Coulomb reference MD at {temp}K\n"
                f"units            metal\n"
                f"atom_style       full\n"
                f"bond_style       harmonic\n"
                f"angle_style      harmonic\n"
                f"improper_style   harmonic\n"
                f"boundary         f f f\n\n"
                f"special_bonds    charmm\n"
                f"read_data        {os.path.abspath(original_data)}\n"
                f"special_bonds    charmm\n\n"
                f"{INFANTE_LAMMPS}\n"
                f"neighbor         2.0 bin\n"
                f"neigh_modify     delay 0 every 1 check yes\n"
                f"timestep         0.001\n\n"
                f"thermo           100\n"
                f"thermo_style     custom step temp pe ke etotal press\n"
                f"log              {infante_log}\n\n"
                f"dump             1 all custom 1 {infante_xyz} element xu yu zu\n"
                f"dump_modify      1 element Cd Se C H O\n"
                f"dump_modify      1 sort id\n\n"
                f"velocity         all create {temp} 12345\n"
                f"fix              1 all nve\n"
                f"fix              2 all temp/csvr {temp}.0 {temp}.0 0.1 54321\n\n"
                f"run              {nsteps}\n")

    print(f"  Running Infante MD ({nsteps} steps at {temp}K)...")
    try:
        result = subprocess.run(
            [lammps_bin, '-in', infante_in],
            cwd=infante_dir, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        print(f"  [Error] LAMMPS binary '{lammps_bin}' not found.")
        return None, None
    if result.returncode != 0:
        print(f"  [Error] Infante MD failed:\n{result.stderr[-1000:]}")
        return None, None

    print(f"  Infante MD complete.")
    return infante_xyz, infante_log


# ============================================================
# PLOTTING
# ============================================================

def plot_rdf(rdf_data, output_file, rdf_pairs):
    print("  Generating RDF plot...")
    pair_keys = [f"{a}-{b}" for a, b in rdf_pairs]
    fig, axes = plt.subplots(len(pair_keys), 1, figsize=(10, 3.5 * len(pair_keys)),
                              sharex=True)
    axes = np.atleast_1d(axes)
    labels = list(rdf_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    # Assign colors: DFT=black, Infante=red, GNN-Morse=steelblue
    color_map = {}
    ci = 0
    for label in labels:
        if 'DFT' in label:
            color_map[label] = 'black'
        elif 'Infante' in label:
            color_map[label] = 'tab:red'
        else:
            color_map[label] = 'steelblue'
            ci += 1

    for ax, pair in zip(axes, pair_keys):
        for label, pair_data in rdf_data.items():
            if pair in pair_data:
                r, gr = pair_data[pair]
                c = color_map.get(label, 'steelblue')
                if 'DFT' in label:
                    ax.plot(r, gr, color='black', ls='--', lw=2, label=label, zorder=100)
                    ax.fill_between(r, gr, alpha=0.05, color='black')
                elif 'Infante' in label:
                    ax.plot(r, gr, color=c, lw=1.8, label=label, alpha=0.85, zorder=50)
                else:
                    ax.plot(r, gr, color=c, lw=1.5, label=label, alpha=0.9)
        ax.set_ylabel("g(r)")
        ax.set_xlim(0, RDF_R_MAX)
        ax.text(0.95, 0.85, pair, transform=ax.transAxes, ha='right',
                fontweight='bold', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("r (A)")
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=min(5, len(labels)), fontsize=9)
    fig.suptitle("Radial Distribution Functions", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.08)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")


def plot_structural(struct_data, output_file):
    print("  Generating structural plot...")
    labels = list(struct_data.keys())
    n_plots = len(labels)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5),
                              sharex=True, sharey=True, constrained_layout=True)
    if n_plots == 1:
        axes = [axes]
    max_count = max(len(struct_data[l]['r']) for l in labels if len(struct_data[l]['r']) > 0)
    max_power = int(np.ceil(np.log10(max(max_count, 1))))

    hb = None
    for ax, label in zip(axes, labels):
        r = struct_data[label]['r']
        theta = struct_data[label]['theta']
        ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'Bond Length $r_{Cd-Se}$ ($\AA$)', fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel(r'Bond Angle $\theta_{Se-Cd-Se}$ ($^\circ$)', fontsize=12)
        if len(r) == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            continue
        hb = ax.hexbin(r, theta, gridsize=80, cmap='viridis',
                       extent=[2.0, 3.5, 60, 180],
                       bins='log', mincnt=1, linewidths=0,
                       vmin=1, vmax=10 ** max_power)
    if hb is not None:
        cbar = fig.colorbar(hb, ax=axes, orientation='vertical',
                            fraction=0.02, pad=0.02)
        cbar.set_label('Count (log scale)', fontsize=12)
        cbar.set_ticks([10**i for i in range(max_power + 1)])
    fig.suptitle("Structural Phase Space Density (Cd-Se)", fontsize=16, fontweight='bold')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"    Saved: {output_file}")


def plot_energy_stability(log_data, output_file):
    print("  Generating energy stability plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    color_map = {}
    for name in log_data:
        if 'Infante' in name:
            color_map[name] = 'tab:red'
        else:
            color_map[name] = 'steelblue'

    for name, data in log_data.items():
        time_ps = data['step'] * 0.001
        c = color_map.get(name, 'steelblue')
        ax1.plot(time_ps, data['pe'], color=c, lw=1.5, label=name, alpha=0.85)
        ax2.plot(time_ps, data['temp'], color=c, lw=1.5, label=name, alpha=0.85)
    ax1.set_ylabel("Potential Energy (eV)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Temperature (K)")
    ax2.set_xlabel("Time (ps)")
    ax2.axhline(300, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("MD Energy Stability", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")


def plot_potentials(cluster_params, output_file):
    """Plot cluster sub-type Morse curves + Infante LJ+Coul reference.

    One subplot per base pair (Cd-Se, Cd-Cd, Se-Se, Cd-O, Se-O).
    Each cluster sub-type gets its own curve; fallbacks are dimmed.
    """
    print("  Generating potential curves (sub-types + Infante)...")
    base_pairs = ['Cd-Se', 'Cd-Cd', 'Se-Se', 'Cd-O', 'O-Se']

    # Only show pairs that have data
    active_pairs = []
    for bp in base_pairs:
        e1, e2 = bp.split('-')
        i, j = sorted((ATOM_MAP.get(e1, 0), ATOM_MAP.get(e2, 0)))
        pk = f'pair_{i}_{j}'
        has_cluster = bp in cluster_params and len(cluster_params[bp]) > 0
        has_infante = pk in INFANTE_REF
        if has_cluster or has_infante:
            active_pairs.append(bp)
    if not active_pairs:
        print("    No potential data.")
        return

    r = np.linspace(0.5, 10.0, 500)
    fig, axes = plt.subplots(len(active_pairs), 1, figsize=(12, 3.5 * len(active_pairs)))
    axes = np.atleast_1d(axes)
    cmap = plt.cm.tab10

    for ax, bp in zip(axes, active_pairs):
        e1, e2 = bp.split('-')
        i, j = sorted((ATOM_MAP.get(e1, 0), ATOM_MAP.get(e2, 0)))
        pk = f'pair_{i}_{j}'

        # Separate real vs fallback cluster pairs
        real_variants = []
        fallback_variants = []
        if bp in cluster_params:
            for name, D, a, r0, is_fb in cluster_params[bp]:
                if is_fb:
                    fallback_variants.append((name, D, a, r0))
                else:
                    real_variants.append((name, D, a, r0))
        # Sort by D_e (strongest first)
        real_variants.sort(key=lambda x: -x[1])

        # Infante LJ+Coul curve
        infante_v = None
        if pk in INFANTE_REF:
            _, eps, sig = INFANTE_REF[pk]
            sig_r = sig / r
            v_lj = 4.0 * eps * (sig_r**12 - sig_r**6)
            q_i = INFANTE_REF.get(f'q_{i}', 0)
            q_j = INFANTE_REF.get(f'q_{j}', 0)
            v_coul = (COULOMB_CONST * q_i * q_j) / r
            infante_v = v_lj + v_coul

        # Compute y-limits from Morse wells
        all_depths = [D for _, D, _, _ in real_variants]
        if all_depths:
            max_D = max(all_depths)
            v_min = -max_D * 1.5
            v_max = max_D * 1.5
        else:
            v_min, v_max = -0.5, 0.5

        # Extend to show Infante well bottom
        if infante_v is not None:
            r_reasonable = r > 2.0
            inf_min = np.min(infante_v[r_reasonable])
            if inf_min < v_min:
                v_min = inf_min * 1.15
                v_max = max(v_max, abs(v_min) * 0.4)

        # Plot Infante (dashed black, background)
        if infante_v is not None:
            v_display = np.clip(infante_v, v_min - 1, v_max + 1)
            ax.plot(r, v_display, color='black', lw=2.5, ls='--',
                    label='Infante (LJ+Coul)', zorder=10, alpha=0.8)

        # Plot real cluster sub-types
        for idx, (name, D, a, r0) in enumerate(real_variants):
            v = D * (1 - np.exp(-a * (r - r0)))**2 - D
            v_display = np.clip(v, v_min - 1, v_max + 1)
            # Shorten label: "Cd_0-Se_2" -> just the cluster indices
            label = name
            ax.plot(r, v_display, color=cmap(idx % 10), lw=1.3,
                    label=label, zorder=1, alpha=0.85)

        ax.axhline(0, color='gray', lw=1.2, alpha=0.4)
        ax.set_xlim(1.5, 7.0)
        ax.set_ylim(v_min, v_max)
        ax.set_ylabel("V (eV)")
        n_real = len(real_variants)
        n_fb = len(fallback_variants)
        title = f"{bp}  ({n_real} sub-types"
        if n_fb:
            title += f", {n_fb} fallback"
        title += ")"
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right', ncol=2,
                  handlelength=1.5, columnspacing=1.0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("r (Angstrom)")
    fig.suptitle("Potential Energy Curves: Cluster Sub-Types + Infante",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")


def plot_force_rerun(dft_forces, lammps_forces, symbols, core_elements, output_file):
    """Force parity + per-frame RMSE plot."""
    print("  Generating force rerun plot...")
    core_mask = np.isin(symbols, list(core_elements))
    dft_f = dft_forces[:, core_mask]
    lmp_f = lammps_forces[:, core_mask]

    diff = dft_f - lmp_f
    per_frame = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    overall = np.sqrt(np.mean(diff ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    dft_flat = dft_f.reshape(-1)
    lmp_flat = lmp_f.reshape(-1)
    if len(dft_flat) > 50000:
        idx = np.random.choice(len(dft_flat), 50000, replace=False)
        dft_flat = dft_flat[idx]
        lmp_flat = lmp_flat[idx]
    ax1.scatter(dft_flat, lmp_flat, s=0.3, alpha=0.15, c='steelblue', rasterized=True)
    lim = max(abs(dft_flat).max(), abs(lmp_flat).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.5)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("DFT Force (eV/A)")
    ax1.set_ylabel("LAMMPS Force (eV/A)")
    ax1.set_title(f"Force Parity (core, RMSE={overall:.4f} eV/A)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    ax2.plot(per_frame, lw=1.0, color='steelblue', alpha=0.8)
    ax2.axhline(overall, color='red', ls='--', lw=1.5,
                label=f'Overall RMSE = {overall:.4f} eV/A')
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("F_RMSE (eV/A)")
    ax2.set_title(f"Per-Frame Force RMSE ({len(per_frame)} frames)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("LAMMPS Force Rerun vs DFT", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")
    return overall


# ============================================================
# FORCE RERUN
# ============================================================

def run_force_rerun(lammps_dir, dft_xyz, core_elements, results_dir, lammps_bin='lmp'):
    """Feed DFT positions through LAMMPS potential, compare forces."""
    import ase.io

    mapping_file = os.path.join(lammps_dir, 'atom_type_mapping.txt')
    data_file = os.path.join(lammps_dir, 'gnn_morse.data')
    potentials_file = os.path.join(lammps_dir, 'potentials.txt')
    lammps_in = os.path.join(lammps_dir, 'gnn_morse.in')

    if not os.path.exists(mapping_file):
        print("  [Skip] No atom_type_mapping.txt found")
        return
    if not os.path.exists(dft_xyz):
        print(f"  [Skip] DFT trajectory not found: {dft_xyz}")
        return

    print(f"\n  Loading DFT trajectory: {os.path.basename(dft_xyz)}")
    dft_frames = ase.io.read(dft_xyz, index=':', format='extxyz')
    dft_symbols = [a.symbol for a in dft_frames[0]]

    data = np.loadtxt(mapping_file, dtype=str, comments='#')
    type_ids = data[:, 1].astype(int)
    cluster_types = sorted(set(data[:, 2]))
    n_atoms = len(type_ids)
    print(f"  Atoms: {n_atoms}, DFT frames: {len(dft_frames)}")

    rerun_dir = os.path.join(results_dir, 'rerun')
    os.makedirs(rerun_dir, exist_ok=True)

    # Write DFT positions as LAMMPS dump
    dft_dump = os.path.join(rerun_dir, 'dft_positions.dump')
    dft_forces_list = []
    with open(dft_dump, 'w') as f:
        for step, frame in enumerate(dft_frames):
            pos = frame.positions
            try:
                dft_forces_list.append(frame.get_forces())
            except Exception:
                dft_forces_list.append(np.zeros((n_atoms, 3)))
            f.write(f"ITEM: TIMESTEP\n{step}\n"
                    f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n"
                    f"ITEM: BOX BOUNDS pp pp pp\n"
                    f"-50.0 50.0\n-50.0 50.0\n-50.0 50.0\n"
                    f"ITEM: ATOMS id type x y z\n")
            for i in range(n_atoms):
                f.write(f"{i+1} {type_ids[i]} "
                        f"{pos[i,0]:.10f} {pos[i,1]:.10f} {pos[i,2]:.10f}\n")
    dft_forces_arr = np.stack(dft_forces_list)
    print(f"  Wrote {len(dft_frames)} frames to DFT dump")

    # Extract potential commands
    potential_cmds = []
    if os.path.exists(lammps_in):
        with open(lammps_in) as f:
            for line in f:
                s = line.strip()
                if s.startswith(('pair_style', 'pair_coeff', 'pair_modify')):
                    potential_cmds.append(s)
                elif s.startswith('include') and 'potentials.txt' in s:
                    if os.path.exists(potentials_file):
                        with open(potentials_file) as pf:
                            for pl in pf:
                                ps = pl.strip()
                                if ps.startswith(('pair_style', 'pair_coeff', 'pair_modify')):
                                    potential_cmds.append(ps)

    if not potential_cmds:
        print("  [Skip] No potential commands found")
        return

    atom_style = 'full'
    style_cmds = []
    if os.path.exists(lammps_in):
        with open(lammps_in) as f:
            for line in f:
                s = line.strip()
                if s.startswith('atom_style'):
                    atom_style = s.split()[-1]
                elif s.startswith(('bond_style', 'angle_style', 'improper_style')):
                    style_cmds.append(s)

    elem_names = ' '.join(base_element(ce) for ce in cluster_types)
    rerun_dump = os.path.join(rerun_dir, 'rerun_forces.dump')
    rerun_in = os.path.join(rerun_dir, 'rerun.in')

    if atom_style == 'charge':
        styles_block = ""
    elif style_cmds:
        styles_block = '\n'.join(style_cmds)
    else:
        styles_block = "bond_style      none\nangle_style     none\nimproper_style  none"

    with open(rerun_in, 'w') as f:
        f.write(f"units           metal\n"
                f"atom_style      {atom_style}\n"
                f"boundary        s s s\n"
                f"{styles_block}\n\n"
                f"read_data       {data_file}\n\n"
                f"{chr(10).join(potential_cmds)}\n\n"
                f"neighbor         2.0 bin\n"
                f"neigh_modify     delay 0 every 1 check yes\n\n"
                f"dump             out all custom 1 {rerun_dump} id type xu yu zu fx fy fz\n"
                f"dump_modify      out element {elem_names}\n"
                f"dump_modify      out sort id\n\n"
                f"rerun            {dft_dump} dump x y z box no\n")

    print("  Running LAMMPS rerun...")
    try:
        result = subprocess.run(
            [lammps_bin, '-in', rerun_in],
            cwd=rerun_dir, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print(f"  [Error] LAMMPS binary '{lammps_bin}' not found. Skipping rerun.")
        return
    if result.returncode != 0:
        print(f"  [Error] LAMMPS rerun failed:\n{result.stderr[-1000:]}")
        return
    print("  LAMMPS rerun completed")

    with open(rerun_dump) as f:
        lines = f.readlines()
    lines_per_frame = 9 + n_atoms
    n_rerun = len(lines) // lines_per_frame
    data_chunks = []
    for frame in range(n_rerun):
        start = frame * lines_per_frame + 9
        data_chunks.extend(lines[start:start + n_atoms])
    raw = np.loadtxt(io.StringIO(''.join(data_chunks)))
    raw = raw.reshape(n_rerun, n_atoms, -1)
    lammps_forces = raw[:, :, 5:8]

    n_common = min(len(dft_forces_arr), len(lammps_forces))
    overall = plot_force_rerun(
        dft_forces_arr[:n_common], lammps_forces[:n_common],
        dft_symbols, core_elements,
        os.path.join(results_dir, 'force_rerun.png'))

    print(f"\n  Force rerun RMSE (core atoms): {overall:.6f} eV/A")
    return overall


# ============================================================
# MAIN
# ============================================================

def main_from_fitter(run_dir, config_path, lammps_bin='lmp', infante=True,
                     nsteps=10000, temp=300):
    """Entry point called from gnn_morse_fitter after training+export."""
    _run_pipeline(
        run_dir=os.path.abspath(run_dir),
        config_path=os.path.abspath(config_path),
        lammps_bin=lammps_bin,
        skip_md=False,
        skip_rerun=False,
        infante=infante,
        nsteps=nsteps,
        temp=temp,
    )


def _run_pipeline(run_dir, config_path, lammps_bin, skip_md, skip_rerun,
                  infante, nsteps, temp):
    """Core analysis pipeline shared by main() and main_from_fitter()."""
    lammps_dir = os.path.join(run_dir, 'lammps')
    results_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  GNN-Morse Full Analysis")
    print("=" * 60)
    print(f"  Run dir: {run_dir}")
    print(f"  LAMMPS dir: {lammps_dir}")

    # Load config
    box_dims = [27.5, 27.5, 27.5]
    dft_xyz = None
    original_data = None
    core_elements = ['Cd', 'Se']
    rdf_pairs = [('Cd', 'Se'), ('Cd', 'Cd'), ('Se', 'Se'), ('Cd', 'O'), ('Se', 'O')]

    if config_path and os.path.exists(config_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        bs = config.get('training', {}).get('box_size_angstrom', 27.5)
        box_dims = [bs, bs, bs]
        core_elements = config.get('training', {}).get('core_elements', core_elements)
        # Resolve paths relative to config dir
        datasets = config.get('datasets', [])
        if datasets:
            xyz_rel = datasets[0].get('xyz', '')
            dft_xyz = os.path.normpath(os.path.join(config_dir, xyz_rel))
        od = config.get('original_data', '')
        if od:
            original_data = os.path.normpath(os.path.join(config_dir, od))
        print(f"  Box size: {bs} A")
        print(f"  Core elements: {core_elements}")
        if dft_xyz:
            print(f"  DFT trajectory: {dft_xyz}")
        if original_data:
            print(f"  Original .data: {original_data}")
    else:
        print("  [Warning] No config found, using defaults")

    # ── Step 1: Run LAMMPS MD (GNN-Morse + optionally Infante) ──
    lammps_in = os.path.join(lammps_dir, 'gnn_morse.in')
    md_xyz = os.path.join(lammps_dir, 'gnn_morse.xyz')
    md_log = os.path.join(lammps_dir, 'gnn_morse.log')

    infante_xyz = None
    infante_log = None

    if not skip_md:
        print(f"\n{'─'*60}")
        print("  Step 1: Running LAMMPS MD")
        print(f"{'─'*60}")

        # GNN-Morse MD
        if not os.path.exists(lammps_in):
            print(f"\n  [Error] {lammps_in} not found.")
            print("  Run the fitter first: gnn-morse config.yaml")
            return
        print(f"\n  Running GNN-Morse MD ({nsteps} steps at {temp}K)...")
        try:
            result = subprocess.run(
                [lammps_bin, '-in', 'gnn_morse.in'],
                cwd=lammps_dir, capture_output=True, text=True, timeout=3600)
        except FileNotFoundError:
            print(f"  [Error] LAMMPS binary '{lammps_bin}' not found.")
            return
        if result.returncode != 0:
            print(f"  [Error] LAMMPS MD failed:\n{result.stderr[-2000:]}")
            return
        print(f"  GNN-Morse MD complete.")

        # Infante MD
        if infante:
            infante_xyz, infante_log = run_infante_md(
                original_data, results_dir,
                temp=temp, nsteps=nsteps, lammps_bin=lammps_bin)
    else:
        print(f"\n  Skipping MD (--skip-md)")
        # Check for existing Infante trajectory
        existing_inf = os.path.join(results_dir, 'infante', 'infante.xyz')
        existing_inf_log = os.path.join(results_dir, 'infante', 'infante.log')
        if os.path.exists(existing_inf):
            infante_xyz = existing_inf
            infante_log = existing_inf_log

    if not os.path.exists(md_xyz):
        print(f"  [Error] No trajectory found: {md_xyz}")
        return

    # ── Step 2: Analyze MD ──
    print(f"\n{'─'*60}")
    print("  Step 2: Analyzing MD trajectories")
    print(f"{'─'*60}")

    box = box_dims + [90.0, 90.0, 90.0]
    rdf_data = {}
    struct_data = {}
    log_data = {}

    # DFT reference
    dft_frames = []
    if dft_xyz and os.path.exists(dft_xyz):
        print(f"\n  Loading DFT reference...")
        dft_frames = parse_xyz_frames(dft_xyz, skip_frames=0)
        print(f"    Frames: {len(dft_frames)}")
        if dft_frames:
            rdf_data['DFT'] = {}
            for a1, a2 in rdf_pairs:
                rdf_data['DFT'][f"{a1}-{a2}"] = compute_rdf(dft_frames, a1, a2, box=box)
            r, theta = analyze_structure(dft_frames)
            struct_data['DFT'] = {'r': r, 'theta': theta}

    # Truncate MD trajectories to DFT length for fair comparison
    n_dft = len(dft_frames)

    # Infante
    infante_frames = []
    if infante_xyz and os.path.exists(infante_xyz):
        print(f"\n  Loading Infante trajectory...")
        infante_frames = parse_xyz_frames(infante_xyz, skip_frames=SKIP_FRAMES)
        if n_dft > 0 and len(infante_frames) > n_dft:
            print(f"    Truncating {len(infante_frames)} -> {n_dft} frames (matching DFT)")
            infante_frames = infante_frames[:n_dft]
        print(f"    Frames: {len(infante_frames)}")
        if infante_frames:
            rdf_data['Infante'] = {}
            for a1, a2 in rdf_pairs:
                rdf_data['Infante'][f"{a1}-{a2}"] = compute_rdf(
                    infante_frames, a1, a2, box=box)
            r, theta = analyze_structure(infante_frames)
            struct_data['Infante'] = {'r': r, 'theta': theta}
        if infante_log and os.path.exists(infante_log):
            inf_data = parse_lammps_log(infante_log)
            if inf_data:
                log_data['Infante'] = inf_data

    # GNN-Morse
    print(f"\n  Loading GNN-Morse trajectory...")
    md_frames = parse_xyz_frames(md_xyz, skip_frames=SKIP_FRAMES)
    if n_dft > 0 and len(md_frames) > n_dft:
        print(f"    Truncating {len(md_frames)} -> {n_dft} frames (matching DFT)")
        md_frames = md_frames[:n_dft]
    print(f"    Frames: {len(md_frames)}")
    if md_frames:
        rdf_data['GNN-Morse'] = {}
        for a1, a2 in rdf_pairs:
            rdf_data['GNN-Morse'][f"{a1}-{a2}"] = compute_rdf(md_frames, a1, a2, box=box)
        r, theta = analyze_structure(md_frames)
        struct_data['GNN-Morse'] = {'r': r, 'theta': theta}
    if os.path.exists(md_log):
        gnn_data = parse_lammps_log(md_log)
        if gnn_data:
            log_data['GNN-Morse'] = gnn_data

    # Generate plots
    print("\n  Generating plots...")
    if rdf_data:
        plot_rdf(rdf_data, os.path.join(results_dir, 'rdf.png'), rdf_pairs)
    if struct_data:
        plot_structural(struct_data, os.path.join(results_dir, 'structural.png'))
    if log_data:
        plot_energy_stability(log_data, os.path.join(results_dir, 'energy_stability.png'))

    # Potential curves (cluster sub-types from potentials.txt or gnn_morse.in)
    potentials_file = os.path.join(lammps_dir, 'potentials.txt')
    if not os.path.exists(potentials_file):
        potentials_file = os.path.join(lammps_dir, 'gnn_morse.in')
    cluster_params, _ = parse_cluster_potentials(potentials_file)
    if cluster_params:
        plot_potentials(cluster_params, os.path.join(results_dir, 'potentials.png'))

    # ── Step 3: Force rerun ──
    if not skip_rerun:
        print(f"\n{'─'*60}")
        print("  Step 3: LAMMPS Force Rerun")
        print(f"{'─'*60}")
        if dft_xyz:
            run_force_rerun(lammps_dir, dft_xyz, core_elements, results_dir,
                            lammps_bin=lammps_bin)
        else:
            print("  [Skip] No DFT trajectory path found")
    else:
        print(f"\n  Skipping force rerun (--skip-rerun)")

    # Summary
    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {results_dir}/")
    for f in ['rdf.png', 'structural.png', 'energy_stability.png',
              'potentials.png', 'force_rerun.png']:
        path = os.path.join(results_dir, f)
        if os.path.exists(path):
            print(f"    - {f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run MD + full analysis')
    parser.add_argument('run_dir', nargs='?', default='.',
                        help='Run directory (default: current dir)')
    parser.add_argument('--skip-md', action='store_true',
                        help='Skip MD, just analyze existing trajectory')
    parser.add_argument('--skip-rerun', action='store_true',
                        help='Skip force rerun')
    parser.add_argument('--no-infante', action='store_true',
                        help='Skip Infante reference MD')
    parser.add_argument('--lammps-bin', default=None,
                        help='Path to LAMMPS binary (default: $LAMMPS_BIN or lmp)')
    parser.add_argument('--nsteps', type=int, default=10000,
                        help='MD steps for both GNN-Morse and Infante (default: 10000)')
    parser.add_argument('--temp', type=int, default=300,
                        help='Temperature in K (default: 300)')
    args = parser.parse_args()

    lammps_bin = args.lammps_bin or os.environ.get('LAMMPS_BIN', 'lmp')
    run_dir = os.path.abspath(args.run_dir)
    config_path = os.path.join(run_dir, 'config.yaml')

    _run_pipeline(
        run_dir=run_dir,
        config_path=config_path,
        lammps_bin=lammps_bin,
        skip_md=args.skip_md,
        skip_rerun=args.skip_rerun,
        infante=not args.no_infante,
        nsteps=args.nsteps,
        temp=args.temp,
    )


if __name__ == '__main__':
    main()
