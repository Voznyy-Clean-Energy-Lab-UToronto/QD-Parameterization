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


def canonical_pair(elem_a, elem_b):
    if elem_a <= elem_b:
        return f"{elem_a}-{elem_b}"
    return f"{elem_b}-{elem_a}"


def base_element(cluster_type):
    return cluster_type.partition('_')[0]


MASSES = {'C': 12.011, 'Cd': 112.411, 'H': 1.008, 'O': 15.999, 'Se': 78.971}

FROZEN_LJ = {
    'C-C':  {'eps': 0.00135800, 'sigma': 3.20724},
    'C-H':  {'eps': 0.00035300, 'sigma': 3.02015},
    'C-O':  {'eps': 0.00142800, 'sigma': 3.20724},
    'H-H':  {'eps': 0.00009200, 'sigma': 2.84197},
    'H-O':  {'eps': 0.00037200, 'sigma': 3.02015},
    'O-O':  {'eps': 0.00150200, 'sigma': 3.20724},
}

# Per-base-pair cutoffs matching training edge distances
BASE_PAIR_CUTOFFS = {
    'Cd-Se': 3.5,
    'Cd-Cd': 7.0,
    'Se-Se': 7.5,
    'C-Cd':  4.5,
    'C-Se':  4.5,
    'Cd-H':  4.5,
    'Cd-O':  4.5,
    'H-Se':  4.5,
    'O-Se':  4.5,
}



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


def has_bonds_in_data(filepath):
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.endswith('bonds') and not stripped.startswith('#'):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return int(parts[0]) > 0
    return False
