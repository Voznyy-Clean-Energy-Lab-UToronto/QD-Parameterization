import torch
from ase.units import Hartree, Bohr

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr


def inverse_softplus(x):
    return torch.log(torch.expm1(torch.as_tensor(x, dtype=torch.float64)))


def canonical_pair(a, b):
    return f"{a}-{b}" if a <= b else f"{b}-{a}"
