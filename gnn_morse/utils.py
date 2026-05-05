import torch
from ase.units import Hartree, Bohr

# unit conversions (internal: Hartree + Bohr)
HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr


def inverse_softplus(x):
    #inverse of softplus: log(exp(x) - 1)
    return torch.log(torch.expm1(torch.as_tensor(x, dtype=torch.float64)))


def canonical_pair(a, b):
    return f"{a}-{b}" if a <= b else f"{b}-{a}"


def base_element(cluster_type):
    return cluster_type.partition('_')[0]

MASSES = {'C': 12.011, 'Cd': 112.411, 'H': 1.008, 'O': 15.999, 'Se': 78.971}
ORGANIC_ELEMENTS = {'C', 'H', 'O'}

# CHARMM36 LJ for organic-organic pairs (formate ligands)
CHARMM_LJ = {
    'C-C':  {'epsilon': 0.003035, 'sigma': 3.564},
    'C-H':  {'epsilon': 0.002020, 'sigma': 2.905},
    'C-O':  {'epsilon': 0.003974, 'sigma': 3.296},
    'H-H':  {'epsilon': 0.001344, 'sigma': 2.245},
    'H-O':  {'epsilon': 0.002645, 'sigma': 2.637},
    'O-O':  {'epsilon': 0.005203, 'sigma': 3.028},
}
