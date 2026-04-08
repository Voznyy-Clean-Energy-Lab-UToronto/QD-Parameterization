import torch
import numpy as np
from ase.units import Hartree, Bohr

# Unit conversions
HARTREE_TO_EV = Hartree          # 27.2114 eV/Ha
BOHR_TO_ANGSTROM = Bohr          # 0.529177 Ang/Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr


def inverse_softplus(x):
    if isinstance(x, (int, float)):
        return float(np.log(np.expm1(x)))
    return torch.log(torch.expm1(x))

# makes sure Cd-Se and Se-Cd are treated as the same thing.
def canonical_pair(elem_a, elem_b):
    if elem_a <= elem_b:
        return f"{elem_a}-{elem_b}"
    return f"{elem_b}-{elem_a}" 

#remove the labelling, so it's just Cd not Cd_core_Se3 for example
def base_element(cluster_type):
    return cluster_type.partition('_')[0]


MASSES = {'C': 12.011, 'Cd': 112.411, 'H': 1.008, 'O': 15.999, 'Se': 78.971}

# CHARMM36 LJ parameters for organic-organic pairs (formate ligands).
CHARMM_LJ = {
    'C-C':  {'epsilon': 0.003035, 'sigma': 3.564},
    'C-H':  {'epsilon': 0.002020, 'sigma': 2.905},
    'C-O':  {'epsilon': 0.003974, 'sigma': 3.296},
    'H-H':  {'epsilon': 0.001344, 'sigma': 2.245},
    'H-O':  {'epsilon': 0.002645, 'sigma': 2.637},
    'O-O':  {'epsilon': 0.005203, 'sigma': 3.028},
}

ORGANIC_ELEMENTS = {'C', 'H', 'O'}

# Reading LAMMPS data file for use in lammps_export.py
def detect_atom_style(filepath):
    with open(filepath) as f:
        for line in f:
            if 'atom_style charge' in line.lower():
                return 'charge'
            if 'atom_style full' in line.lower() or 'Full Topology' in line:
                return 'full'
            stripped = line.strip()
            if stripped.endswith('bonds') and not stripped.startswith('#'):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0] == '0':
                    return 'charge'
                elif parts[0].isdigit() and int(parts[0]) > 0:
                    return 'full'
    # Fallback: check Atoms section column count
    in_atoms = False
    with open(filepath) as f:
        for line in f:
            if line.strip() == 'Atoms' or line.strip().startswith('Atoms #'):
                in_atoms = True
                continue
            if in_atoms and line.strip():
                return 'full' if len(line.split()) >= 7 else 'charge'
    return 'full'

# To determine if the LAMMPS data file has bonds or not (related to filling in CHARMM details). also for lammps_export.py
def has_bonds_in_data(filepath):
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.endswith('bonds') and not stripped.startswith('#'):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return int(parts[0]) > 0
    return False
