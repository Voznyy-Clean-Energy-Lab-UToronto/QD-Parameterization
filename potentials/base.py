"""Abstract base class for pair potentials."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


def inverse_softplus(x):
    """Inverse of softplus: returns y such that softplus(y) = x."""
    return torch.log(torch.exp(torch.as_tensor(x, dtype=torch.float64)) - 1.0 + 1e-20)


class BasePotential(ABC, nn.Module):
    """Base class for all pair potentials.

    Each potential defines the short-range (non-Coulomb) pairwise interaction.
    Parameters are stored as per-pair tensors of shape (num_pairs,).
    """

    def __init__(self, num_pairs, initial_values=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.create_parameters(num_pairs, initial_values or {})

    # ── Parameter interface ──────────────────────────────────────────

    @staticmethod
    @abstractmethod
    def user_param_names():
        """User-facing parameter names for YAML config.
        e.g. ['epsilon', 'sigma'] for LJ, ['D_e', 'alpha', 'r0'] for Morse.
        """

    @staticmethod
    def param_kinds():
        """Return dict mapping user_param_name -> unit kind string.
        Kinds: 'energy', 'distance', 'inv_distance', 'dimensionless',
               'energy_dist6', 'energy_dist8', 'energy_dist' (energy*distance).
        Used by the fitter for unit conversion of YAML-provided values.
        Override in subclasses.
        """
        return {}

    @staticmethod
    @abstractmethod
    def parameter_names():
        """Raw nn.Parameter attribute names stored on this module.
        e.g. ['raw_epsilon', 'raw_sigma'] for LJ.
        """

    @abstractmethod
    def create_parameters(self, num_pairs, initial_values):
        """Create nn.Parameters from user-facing initial values dict.

        Args:
            num_pairs: number of pair types
            initial_values: dict mapping user_param_name -> np.array of shape (num_pairs,)
        """

    @abstractmethod
    def get_constrained_params(self):
        """Return dict of user_param_name -> constrained tensor (num_pairs,).
        Applies softplus or other constraints to raw parameters.
        """

    # ── Physics ──────────────────────────────────────────────────────

    @staticmethod
    @abstractmethod
    def required_edge_fields():
        """Edge-level precomputed fields needed by this potential.
        Always available: 'inverse_distances', 'inverse_distances_sq' (for Coulomb).
        Return additional fields needed, e.g. ['distances'] for Morse,
        or ['inverse_distances_6', 'inverse_distances_12'] for LJ.
        """

    @abstractmethod
    def scalar_force(self, batch, pair_indices):
        """Compute scalar force -dV/dr for each edge.

        Args:
            batch: PyG Data/Batch with precomputed edge fields
            pair_indices: (num_edges,) tensor of pair-type indices

        Returns:
            (num_edges,) tensor of scalar forces (positive = repulsive)
        """

    # ── Equilibrium / RDF initialization ─────────────────────────────

    @staticmethod
    @abstractmethod
    def equilibrium_distance_param():
        """Which user param maps to RDF peak distance.
        For LJ: 'sigma' (peak = sigma * 2^(1/6)).
        For Morse/Buck: 'r0' (peak = r0 directly).
        """

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """Convert RDF peak distance to the equilibrium distance parameter value.
        Override in subclasses. Default: param = peak (works for Morse, Buck, etc.).
        """
        return peak_distance

    # ── LAMMPS export ────────────────────────────────────────────────

    @abstractmethod
    def lammps_pair_style(self, cutoff_angstrom):
        """LAMMPS pair_style command string, or None if tabulated."""

    @abstractmethod
    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        """List of LAMMPS pair_coeff lines.

        Args:
            pair_names: list of 'El1-El2' strings
            elements: list of element strings (for LAMMPS type numbering)
            cutoff_angstrom: cutoff in Angstroms

        Returns:
            list of strings (each a pair_coeff command)
        """

    # ── Display ──────────────────────────────────────────────────────

    @abstractmethod
    def get_params_display(self):
        """Return dict of user_param_name -> np.array in physical units (eV, Angstrom).
        Used for printing and saving.
        """

    def display_header(self):
        """Column header string for parameter printing."""
        names = self.user_param_names()
        return " | ".join(f"{n:>12}" for n in names)

    def display_row(self, pair_index):
        """Formatted parameter values for one pair."""
        params = self.get_params_display()
        return " | ".join(
            f"{params[name][pair_index]:12.6f}" for name in self.user_param_names()
        )

    # ── Frozen mask ──────────────────────────────────────────────────

    def apply_frozen_mask(self, frozen_mask):
        """Zero out gradients for frozen pairs. Works for any number of params."""
        if frozen_mask is None:
            return
        for pname in self.parameter_names():
            param = getattr(self, pname)
            if param.grad is not None:
                param.grad *= frozen_mask

    # ── Table export utility (for potentials without native LAMMPS style) ─

    def generate_table_file(self, pair_name, cutoff_angstrom, num_points=5000,
                            r_min_angstrom=0.5):
        """Generate a LAMMPS-compatible pair_style table file.

        Returns:
            string contents of the table file
        """
        from ase.units import Hartree, Bohr
        BOHR_TO_ANG = Bohr
        ANG_TO_BOHR = 1.0 / Bohr
        HARTREE_TO_EV = Hartree

        r_ang = np.linspace(r_min_angstrom, cutoff_angstrom, num_points)
        r_bohr = r_ang * ANG_TO_BOHR

        # Build a fake batch with just distances for this pair
        pair_idx = 0  # we'll use the first pair slot
        constrained = self.get_constrained_params()

        # We need to evaluate V(r) and f(r) = -dV/dr
        # Use autograd on a single-pair version
        r_tensor = torch.tensor(r_bohr, dtype=torch.float64, requires_grad=True)

        energy = self._pair_energy(r_tensor, pair_idx)
        force_scalar = -torch.autograd.grad(
            energy.sum(), r_tensor, create_graph=False
        )[0]

        # Convert to eV and eV/Angstrom
        e_ev = energy.detach().numpy() * HARTREE_TO_EV
        f_ev_ang = force_scalar.detach().numpy() * HARTREE_TO_EV / BOHR_TO_ANG

        lines = []
        lines.append(f"# Table for {pair_name}")
        lines.append(f"{pair_name}")
        lines.append(f"N {num_points} R {r_min_angstrom} {cutoff_angstrom}")
        lines.append("")
        for i in range(num_points):
            lines.append(f"{i+1} {r_ang[i]:.10f} {e_ev[i]:.10e} {f_ev_ang[i]:.10e}")

        return "\n".join(lines)

    def _pair_energy(self, r_bohr, pair_index):
        """Compute V(r) for a single pair type. Override for table export.
        Default raises NotImplementedError for potentials that don't need it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _pair_energy() for table export"
        )
