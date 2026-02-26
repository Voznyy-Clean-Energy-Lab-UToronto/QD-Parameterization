"""Potential registry and factory function."""

from .lj import LennardJones
from .morse import Morse
from .buckingham import Buckingham
from .mie import Mie
from .yukawa import Yukawa
from .born import BornMayerHuggins
from .rydberg import Rydberg
from .double_exp import DoubleExponential
from .damped_buck import DampedBuckingham
from .soft_core_lj import SoftCoreLJ
from .tersoff import Tersoff
from .stillinger_weber import StillingerWeber

POTENTIAL_REGISTRY = {
    'lj': LennardJones,
    'morse': Morse,
    'buckingham': Buckingham,
    'buck': Buckingham,
    'mie': Mie,
    'yukawa': Yukawa,
    'born': BornMayerHuggins,
    'bmh': BornMayerHuggins,
    'rydberg': Rydberg,
    'double_exp': DoubleExponential,
    'damped_buck': DampedBuckingham,
    'soft_core_lj': SoftCoreLJ,
    'tersoff': Tersoff,
    'sw': StillingerWeber,
    'stillinger_weber': StillingerWeber,
}


def get_potential(name):
    """Look up a potential class by name.

    Args:
        name: string key (e.g. 'lj', 'morse', 'buck')

    Returns:
        BasePotential subclass (not instantiated)

    Raises:
        ValueError: if name is not in the registry
    """
    key = name.lower().strip()
    if key not in POTENTIAL_REGISTRY:
        available = ', '.join(sorted(POTENTIAL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown potential type '{name}'. Available: {available}"
        )
    return POTENTIAL_REGISTRY[key]


def list_potentials():
    """Return sorted list of available potential names."""
    return sorted(set(
        cls.__name__ for cls in POTENTIAL_REGISTRY.values()
    ))
