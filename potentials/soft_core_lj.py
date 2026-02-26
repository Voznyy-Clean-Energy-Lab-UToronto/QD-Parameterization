"""Soft-core Lennard-Jones pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
LJ_EQUILIBRIUM_FACTOR = 2 ** (1 / 6)


class SoftCoreLJ(BasePotential):
    """Soft-core Lennard-Jones pair potential.

    V(r) = 4 * eps * [ (sig^6 / (r^6 + delta^6))^2 - sig^6 / (r^6 + delta^6) ]

    The soft-core modification replaces 1/r^6 with 1/(r^6 + delta^6), which
    removes the singularity at r=0. When delta=0, this reduces to the standard
    12-6 LJ potential. The parameter delta controls the softening: larger delta
    means a softer, more rounded repulsive core.

    This is useful for:
    - Avoiding numerical instabilities when atoms approach closely during fitting
    - Free energy perturbation calculations (though here delta is a fitted parameter)
    - Systems where the standard LJ r^-12 repulsion is too steep

    Force (analytical):
        Let s = sig^6 / (r^6 + delta^6), then V = 4*eps*(s^2 - s).
        ds/dr = -6 * sig^6 * r^5 / (r^6 + delta^6)^2
        dV/dr = 4*eps*(2*s - 1)*ds/dr
        f(r) = -dV/dr = 4*eps*(2*s - 1) * 6*sig^6*r^5 / (r^6 + delta^6)^2

    Parameters (internal, atomic units):
        epsilon: well depth (Hartree), constrained positive via softplus
        sigma:   distance parameter (Bohr), unconstrained
        delta:   soft-core parameter (Bohr), constrained positive via softplus
    """

    @staticmethod
    def user_param_names():
        return ['epsilon', 'sigma', 'delta']

    @staticmethod
    def param_kinds():
        return {'epsilon': 'energy', 'sigma': 'distance', 'delta': 'distance'}

    @staticmethod
    def parameter_names():
        return ['raw_epsilon', 'raw_sigma', 'raw_delta']

    def create_parameters(self, num_pairs, initial_values):
        # epsilon: default 0.01 eV, converted to Hartree, softplus-constrained
        eps_init = torch.tensor(
            initial_values.get('epsilon', np.full(num_pairs, 0.01 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_epsilon = nn.Parameter(inverse_softplus(eps_init))

        # sigma: default 2.5 Angstrom, converted to Bohr, unconstrained
        sig_init = torch.tensor(
            initial_values.get('sigma', np.full(num_pairs, 2.5 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_sigma = nn.Parameter(sig_init)

        # delta: default 0.5 Angstrom, converted to Bohr, softplus-constrained
        delta_init = torch.tensor(
            initial_values.get('delta', np.full(num_pairs, 0.5 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_delta = nn.Parameter(inverse_softplus(delta_init))

    def get_constrained_params(self):
        return {
            'epsilon': nn.functional.softplus(self.raw_epsilon),
            'sigma': self.raw_sigma,
            'delta': nn.functional.softplus(self.raw_delta),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge analytically.

        Let s = sig^6 / (r^6 + delta^6).
        V = 4*eps*(s^2 - s)
        ds/dr = -6 * sig^6 * r^5 / (r^6 + delta^6)^2
        f = -dV/dr = -4*eps*(2*s - 1)*ds/dr
          = 4*eps*(2*s - 1) * 6 * sig^6 * r^5 / (r^6 + delta^6)^2

        This is positive (repulsive) when s > 0.5 (short range, where the
        repulsive s^2 term dominates) and negative (attractive) when s < 0.5
        (long range, where the attractive -s term dominates).
        """
        epsilon = nn.functional.softplus(self.raw_epsilon)
        sigma = self.raw_sigma
        delta = nn.functional.softplus(self.raw_delta)

        pair_epsilon = epsilon[pair_indices]
        pair_sigma = sigma[pair_indices]
        pair_delta = delta[pair_indices]

        distances = batch.distances

        sigma_6 = pair_sigma ** 6
        delta_6 = pair_delta ** 6
        r_6 = distances ** 6
        r_5 = distances ** 5

        denominator = r_6 + delta_6
        denominator_sq = denominator ** 2

        # s = sig^6 / (r^6 + delta^6)
        s = sigma_6 / denominator

        # ds/dr = -6 * sig^6 * r^5 / (r^6 + delta^6)^2
        ds_dr = -6.0 * sigma_6 * r_5 / denominator_sq

        # f = -dV/dr = -4*eps*(2s - 1)*ds/dr
        return -4.0 * pair_epsilon * (2.0 * s - 1.0) * ds_dr

    @staticmethod
    def equilibrium_distance_param():
        return 'sigma'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For SoftCoreLJ: sigma = peak / 2^(1/6), same as standard LJ.

        The equilibrium distance shifts slightly with nonzero delta, but
        sigma / 2^(1/6) remains a good initialization.
        """
        return peak_distance / LJ_EQUILIBRIUM_FACTOR

    def lammps_pair_style(self, cutoff_angstrom):
        """No native LAMMPS pair_style for soft-core LJ; use tabulated."""
        return None

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        """Generate pair_coeff lines for LAMMPS pair_style table."""
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            table_keyword = pair_name.replace('-', '_')
            lines.append(
                f"pair_coeff {t1} {t2} table_file.dat {table_keyword}"
            )
        return lines

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'epsilon': params['epsilon'].detach().cpu().numpy() * HARTREE_TO_EV,
            'sigma': params['sigma'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'delta': params['delta'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = 4*eps*[(sig^6/(r^6+delta^6))^2 - sig^6/(r^6+delta^6)]."""
        epsilon = nn.functional.softplus(self.raw_epsilon)[pair_index]
        sigma = self.raw_sigma[pair_index]
        delta = nn.functional.softplus(self.raw_delta)[pair_index]

        sigma_6 = sigma ** 6
        delta_6 = delta ** 6
        r_6 = r_bohr ** 6

        # s = sig^6 / (r^6 + delta^6)
        s = sigma_6 / (r_6 + delta_6)

        return 4.0 * epsilon * (s ** 2 - s)
