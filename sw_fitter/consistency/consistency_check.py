import os
import shutil
import subprocess

import numpy as np
import torch

from ..data import DFTDataset
from ..fitter import initialise_parameters, load_config
from ..lammps_export import export_lammps
from ..models import sw_forces
from ..utils import BOHR_TO_ANGSTROM, FORCE_AU_TO_EV_ANG

MASSES = {"C": 12.011, "Cd": 112.411, "H": 1.008, "O": 15.999, "Se": 78.971}


def write_lammps_data(filepath, subtypes, elements, positions_ang):

    type_of = {element: i + 1 for i, element in enumerate(elements)}
    low = positions_ang.min(axis=0) - 15.0
    high = positions_ang.max(axis=0) + 15.0
    with open(filepath, "w") as f:
        f.write("SW consistency check\n\n")
        f.write(f"{len(subtypes)} atoms\n{len(elements)} atom types\n\n")
        for axis, name in enumerate("xyz"):
            f.write(f"{low[axis]:.4f} {high[axis]:.4f} {name}lo {name}hi\n")
        f.write("\nMasses\n\n")
        for element in elements:
            f.write(f"{type_of[element]} {MASSES[element.split('_')[0]]:.4f}  # {element}\n")
        f.write("\nAtoms\n\n")
        for atom, subtype in enumerate(subtypes):
            x, y, z = positions_ang[atom]
            f.write(f"{atom + 1} {type_of[subtype]} 0.0 {x:.6f} {y:.6f} {z:.6f}\n")


def write_lammps_input(filepath, data_filename, elements, sw_elements, sw_filename):

    mapping = " ".join(e if e in sw_elements else "NULL" for e in elements)
    with open(filepath, "w") as f:
        f.write("units metal\natom_style charge\nboundary s s s\n")
        f.write(f"read_data {data_filename}\n")
        f.write("pair_style hybrid/overlay sw zero 6.0\n")
        f.write("pair_coeff * * zero\n")
        f.write(f"pair_coeff * * sw {sw_filename} {mapping}\n")
        f.write("neighbor 2.0 bin\nneigh_modify delay 0 every 1 check yes\n")
        f.write("thermo 1\nthermo_style custom step pe\n")
        f.write("dump 1 all custom 1 forces.dump id fx fy fz\n")
        f.write("dump_modify 1 sort id\nrun 0\n")


def read_lammps_forces(filepath, n_atoms):
    forces = np.zeros((n_atoms, 3))
    lines = open(filepath).read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")) + 1
    for line in lines[start:]:
        columns = line.split()
        if len(columns) >= 4:
            forces[int(columns[0]) - 1] = [float(c) for c in columns[1:4]]
    return forces


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="sw-fitter-check",
        description="Compare the initial Python SW forces with a LAMMPS export"
    )
    parser.add_argument("config", help="config YAML used by sw-fitter")
    parser.add_argument("--lammps", default=None, help="path to the LAMMPS executable")
    parser.add_argument("--frames", type=int, default=80, help="frames loaded per dataset")
    parser.add_argument("--output-dir", default="consistency_results")
    args = parser.parse_args()

    lammps_bin = args.lammps or shutil.which("lmp")
    if lammps_bin is None:
        parser.error("LAMMPS was not found; pass its executable with --lammps")
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    config = load_config(args.config)
    for dataset_config in config["datasets"]:
        dataset_config["first_n_frames"] = args.frames

    data = DFTDataset(config["datasets"], config["scope"])
    data.build_graphs()
    params = initialise_parameters(data, config["training"].get("training_temperature", 650),
                                   config.get("eps_init", {}), config.get("eps_init_default", 0.01))
    graph = data.graphs[0]

    with torch.no_grad():
        sw_force = sw_forces(graph, params).numpy() * FORCE_AU_TO_EV_ANG

    positions_ang = graph["positions"].numpy() * BOHR_TO_ANGSTROM
    symbols = data.datasets[0]["symbols"]
    sw_filename = f"{data.chemical_formula}.sw"
    write_lammps_data(
        os.path.join(output_dir, "config.data"), symbols, data.elements, positions_ang
    )
    sw_elements = export_lammps(output_dir, data, params, data.chemical_formula)
    write_lammps_input(os.path.join(output_dir, "force_eval.in"), "config.data",
                       data.elements, sw_elements, sw_filename)

    print("\nrunning LAMMPS...")
    result = subprocess.run([lammps_bin, "-in", "force_eval.in"],
                            cwd=output_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print((result.stdout + result.stderr)[-2000:])
        raise SystemExit("LAMMPS failed -- see above.")

    lammps_force = read_lammps_forces(
        os.path.join(output_dir, "forces.dump"), len(symbols)
    )
    difference = sw_force - lammps_force
    max_diff = np.abs(difference).max()
    force_scale = np.abs(lammps_force).max()
    print("\n=== SW forces vs LAMMPS (eV/Angstrom) ===")
    print(f"  force scale (max |F|): {force_scale:.4f}")
    print(f"  max |delta F|: {max_diff:.3e}   RMS delta F: "
          f"{np.sqrt((difference**2).mean()):.3e}")
    if max_diff < 1e-4 * max(force_scale, 1e-6):
        print("  PASS: the export reproduces the model.")
    else:
        print("  FAIL: model and LAMMPS disagree. Worst atoms:")
        for atom in np.argsort(-np.abs(difference).max(axis=1))[:8]:
            print(f"    atom {atom} ({symbols[atom]}): "
                  f"model={sw_force[atom]}  lammps={lammps_force[atom]}")


if __name__ == "__main__":
    main()
