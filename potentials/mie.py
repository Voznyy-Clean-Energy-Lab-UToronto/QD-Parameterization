"""Mie (generalized Lennard-Jones) pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr

# Default exponents match the standard 12-6 LJ
DEFAULT_GAMMA_R = 12.0
DEFAULT_GAMMA_A = 6.0
LJ_EQUILIBRIUM_FACTOR = 2 ** (1 / 6)


class Mie(BasePotential):
    """Mie (generalized Lennard-Jones) pair potential.

    V(r) = C_mie * epsilon * [(sigma/r)^n - (sigma/r)^m]

    where the Mie prefactor is:

        C_mie = (n / (n - m)) * (n / m) ^ (m / (n - m))

    This reduces to the standard 12-6 LJ (with C_mie = 4) when n=12, m=6.

    Parameters (internal, atomic units):
        epsilon:  well depth (Hartree), constrained positive via softplus
        sigma:    size parameter (Bohr), unconstrained
        gamma_r:  repulsive exponent n (dimensionless), constrained positive via softplus
        gamma_a:  attractive exponent m (dimensionless), constrained positive via softplus

    The force is:
        f(r) = -dV/dr = C_mie * epsilon * [n * sigma^n / r^(n+1) - m * sigma^m / r^(m+1)]
    """

    @staticmethod
    def user_param_names():
        return ['epsilon', 'sigma', 'gamma_r', 'gamma_a']

    @staticmethod
    def param_kinds():
        return {
            'epsilon': 'energy', 'sigma': 'distance',
            'gamma_r': 'dimensionless', 'gamma_a': 'dimensionless',
        }

    @staticmethod
    def parameter_names():
        return ['raw_epsilon', 'raw_sigma', 'raw_gamma_r', 'raw_gamma_a']

    def create_parameters(self, num_pairs, initial_values):
        # epsilon: default 0.01 eV, converted to Hartree, softplus-constrained
        epsilon_init = torch.tensor(
            initial_values.get('epsilon', np.full(num_pairs, 0.01 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_epsilon = nn.Parameter(inverse_softplus(epsilon_init))

        # sigma: default 2.5 Angstrom, converted to Bohr, unconstrained
        sigma_init = torch.tensor(
            initial_values.get('sigma', np.full(num_pairs, 2.5 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_sigma = nn.Parameter(sigma_init)

        # gamma_r: repulsive exponent, default 12 (dimensionless), softplus-constrained
        gamma_r_init = torch.tensor(
            initial_values.get('gamma_r', np.full(num_pairs, DEFAULT_GAMMA_R)),
            dtype=torch.float64,
        )
        self.raw_gamma_r = nn.Parameter(inverse_softplus(gamma_r_init))

        # gamma_a: attractive exponent, default 6 (dimensionless), softplus-constrained
        gamma_a_init = torch.tensor(
            initial_values.get('gamma_a', np.full(num_pairs, DEFAULT_GAMMA_A)),
            dtype=torch.float64,
        )
        self.raw_gamma_a = nn.Parameter(inverse_softplus(gamma_a_init))

    def get_constrained_params(self):
        return {
            'epsilon': nn.functional.softplus(self.raw_epsilon),
            'sigma': self.raw_sigma,
            'gamma_r': nn.functional.softplus(self.raw_gamma_r),
            'gamma_a': nn.functional.softplus(self.raw_gamma_a),
        }

    @staticmethod
    def required_edge_fields():
        # Need actual distances because exponents are variable (cannot precompute)
        return ['distances']

    def _compute_mie_prefactor(self, n, m):
        """Compute C_mie = (n / (n - m)) * (n / m) ^ (m / (n - m)).

        For n=12, m=6 this gives C_mie = 4 (standard LJ).
        """
        n_minus_m = n - m
        return (n / n_minus_m) * (n / m) ** (m / n_minus_m)

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = C_mie * epsilon * [n * sigma^n / r^(n+1) - m * sigma^m / r^(m+1)]

        At short range (r << sigma): the (sigma/r)^n repulsive term dominates,
        giving positive f (repulsive). At long range (r >> sigma): both terms
        vanish, with the attractive term decaying more slowly.
        """
        epsilon = nn.functional.softplus(self.raw_epsilon)
        sigma = self.raw_sigma
        gamma_r = nn.functional.softplus(self.raw_gamma_r)
        gamma_a = nn.functional.softplus(self.raw_gamma_a)

        pair_epsilon = epsilon[pair_indices]
        pair_sigma = sigma[pair_indices]
        pair_n = gamma_r[pair_indices]
        pair_m = gamma_a[pair_indices]

        c_mie = self._compute_mie_prefactor(pair_n, pair_m)
        distances = batch.distances

        # sigma/r ratio raised to the n-th and m-th powers
        sigma_over_r = pair_sigma / distances
        repulsive_term = pair_n * sigma_over_r ** pair_n / distances
        attractive_term = pair_m * sigma_over_r ** pair_m / distances

        return c_mie * pair_epsilon * (repulsive_term - attractive_term)

    @staticmethod
    def equilibrium_distance_param():
        return 'sigma'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Mie with default n=12, m=6: sigma = peak / 2^(1/6).

        This is exact only for 12-6 LJ. For other exponents the equilibrium
        distance shifts, but this provides a reasonable initialization since the
        exponents will also be optimized.
        """
        return peak_distance / LJ_EQUILIBRIUM_FACTOR

    def lammps_pair_style(self, cutoff_angstrom):
        return f"mie/cut {cutoff_angstrom}"

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        """Generate LAMMPS pair_coeff lines.

        LAMMPS mie/cut format: pair_coeff type1 type2 epsilon sigma gamma_r gamma_a
        where epsilon is in eV, sigma in Angstrom, and gammas are dimensionless exponents.
        """
        params = self.get_params_display()
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            eps = params['epsilon'][i]
            sig = params['sigma'][i]
            gr = params['gamma_r'][i]
            ga = params['gamma_a'][i]
            lines.append(
                f"pair_coeff {t1} {t2} {eps:.8f} {sig:.8f} {gr:.8f} {ga:.8f}"
            )
        return lines

    def get_params_display(self):
        """Return parameters in physical units (eV, Angstrom, dimensionless)."""
        params = self.get_constrained_params()
        return {
            'epsilon': params['epsilon'].detach().cpu().numpy() * HARTREE_TO_EV,
            'sigma': params['sigma'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'gamma_r': params['gamma_r'].detach().cpu().numpy(),
            'gamma_a': params['gamma_a'].detach().cpu().numpy(),
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = C_mie * epsilon * [(sigma/r)^n - (sigma/r)^m] for a single pair type."""
        epsilon = nn.functional.softplus(self.raw_epsilon)[pair_index]
        sigma = self.raw_sigma[pair_index]
        n = nn.functional.softplus(self.raw_gamma_r)[pair_index]
        m = nn.functional.softplus(self.raw_gamma_a)[pair_index]

        c_mie = self._compute_mie_prefactor(n, m)
        sigma_over_r = sigma / r_bohr
        return c_mie * epsilon * (sigma_over_r ** n - sigma_over_r ** m)
