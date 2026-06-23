from ase.units import Bohr, Hartree

HARTREE_TO_EV = Hartree  # multiply a Hartree energy to get eV
BOHR_TO_ANGSTROM = Bohr  # multiply a Bohr length to get Angstrom
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr  # Hartree/Bohr -> eV/Angstrom


def canonical_pair(type_a, type_b):
    if type_a <= type_b:
        return f"{type_a}-{type_b}"
    return f"{type_b}-{type_a}"


def canonical_triplet(centre, neighbour_a, neighbour_b):
    first, second = sorted([neighbour_a, neighbour_b])
    return f"{centre}:{first}-{second}"
