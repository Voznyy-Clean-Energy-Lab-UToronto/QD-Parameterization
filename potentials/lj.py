"""Lennard-Jones pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr
LJ_EQUILIBRIUM_FACTOR = 2 ** (1/6)


class LennardJones(BasePotential):
    """Standard 12-6 Lennard-Jones potential.

    V(r) = 4*eps * [(sig/r)^12 - (sig/r)^6]
    f(r) = -dV/dr = 24*eps/r * [2*(sig/r)^12 - (sig/r)^6]
    """

    @staticmethod
    def user_param_names():
        return ['epsilon', 'sigma']

    @staticmethod
    def param_kinds():
        return {'epsilon': 'energy', 'sigma': 'distance'}

    @staticmethod
    def parameter_names():
        return ['raw_epsilon', 'raw_sigma']

    def create_parameters(self, num_pairs, initial_values):
        eps_init = torch.tensor(
            initial_values.get('epsilon', np.full(num_pairs, 0.01 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_epsilon = nn.Parameter(inverse_softplus(eps_init))

        sig_init = torch.tensor(
            initial_values.get('sigma', np.full(num_pairs, 2.5 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_sigma = nn.Parameter(sig_init)

    def get_constrained_params(self):
        return {
            'epsilon': nn.functional.softplus(self.raw_epsilon),
            'sigma': self.raw_sigma,
        }

    @staticmethod
    def required_edge_fields():
        return ['inverse_distances_6', 'inverse_distances_12']

    def scalar_force(self, batch, pair_indices):
        epsilon = nn.functional.softplus(self.raw_epsilon)
        pair_epsilon = epsilon[pair_indices]
        pair_sigma = self.raw_sigma[pair_indices]
        sigma_6 = pair_sigma ** 6
        sigma_12 = sigma_6 ** 2

        return (
            24.0 * pair_epsilon * batch.inverse_distances
            * (2.0 * sigma_12 * batch.inverse_distances_12
               - sigma_6 * batch.inverse_distances_6)
        )

    @staticmethod
    def equilibrium_distance_param():
        return 'sigma'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For LJ: sigma = peak / 2^(1/6)."""
        return peak_distance / LJ_EQUILIBRIUM_FACTOR

    def lammps_pair_style(self, cutoff_angstrom):
        return f"lj/cut {cutoff_angstrom}"

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        params = self.get_params_display()
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            eps = params['epsilon'][i]
            sig = params['sigma'][i]
            lines.append(f"pair_coeff {t1} {t2} {eps:.8f} {sig:.8f}")
        return lines

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'epsilon': params['epsilon'].detach().cpu().numpy() * HARTREE_TO_EV,
            'sigma': params['sigma'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        epsilon = nn.functional.softplus(self.raw_epsilon)[pair_index]
        sigma = self.raw_sigma[pair_index]
        sr6 = (sigma / r_bohr) ** 6
        sr12 = sr6 ** 2
        return 4.0 * epsilon * (sr12 - sr6)
