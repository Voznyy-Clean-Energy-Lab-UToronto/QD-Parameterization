from ase.units import Bohr, Hartree

# Unit conversions used throughout program. Assumes DFT output is in eV/A, converts to atomic units (Ha, Bohr) internally
# for all calculations. Converts back to eV/A for output.
HARTREE_TO_EV = Hartree  # multiply a Hartree energy to get eV
BOHR_TO_ANGSTROM = Bohr  # multiply a Bohr length to get Angstrom
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
FORCE_AU_TO_EV_ANG = Hartree / Bohr  # Hartree/Bohr -> eV/Angstrom


# pairwise interactions are made into a string, with the alphabetically "first" element on the left. It produces the key
# type_a-type_b, to be used in a dict without breaking anything. This prevents type_a-type_b and type_b-type_a being seperate
# interactions, even though they should be identical.
def canonical_pair(type_a, type_b):
    if type_a <= type_b:
        return f"{type_a}-{type_b}"
    return f"{type_b}-{type_a}"


# triplet (many-body) interactions are also made into a string. sorted() prevents meaningless duplicates from occurring.
# The center atom is the "important" one in a physics sense. The legs are interchangable, so O:Cd-Se and O:Se-Cd are identical
# This is just for naming, it does not determine valid triplets itself. That is dependent on the data it recieves (data.py)
def canonical_triplet(centre, neighbour_a, neighbour_b):
    first, second = sorted([neighbour_a, neighbour_b])
    return f"{centre}:{first}-{second}"
