"""Morse pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr


class Morse(BasePotential):
    """Morse pair potential.

    V(r) = D_e * (1 - exp(-alpha * (r - r0)))^2
    f(r) = -dV/dr = 2 * D_e * alpha * [exp(-2*alpha*(r - r0)) - exp(-alpha*(r - r0))]

    Parameters (internal, atomic units):
        D_e:   well depth (Hartree), constrained positive via softplus
        alpha: width parameter (1/Bohr), constrained positive via softplus
        r0:    equilibrium distance (Bohr), unconstrained
    """

    @staticmethod
    def user_param_names():
        return ['D_e', 'alpha', 'r0']

    @staticmethod
    def param_kinds():
        return {'D_e': 'energy', 'alpha': 'inv_distance', 'r0': 'distance'}

    @staticmethod
    def parameter_names():
        return ['raw_D_e', 'raw_alpha', 'raw_r0']

    def create_parameters(self, num_pairs, initial_values):
        # D_e: default 0.1 eV, converted to Hartree, softplus-constrained
        d_e_init = torch.tensor(
            initial_values.get('D_e', np.full(num_pairs, 0.1 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_D_e = nn.Parameter(inverse_softplus(d_e_init))

        # alpha: default 1.5 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # alpha[1/Bohr] = alpha[1/Ang] * BOHR_TO_ANGSTROM (since 1/Ang * Ang/Bohr = 1/Bohr)
        alpha_init = torch.tensor(
            initial_values.get('alpha', np.full(num_pairs, 1.5 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_alpha = nn.Parameter(inverse_softplus(alpha_init))

        # r0: default 3.0 Angstrom, converted to Bohr, unconstrained
        r0_init = torch.tensor(
            initial_values.get('r0', np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_r0 = nn.Parameter(r0_init)

    def get_constrained_params(self):
        return {
            'D_e': nn.functional.softplus(self.raw_D_e),
            'alpha': nn.functional.softplus(self.raw_alpha),
            'r0': self.raw_r0,
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = 2 * D_e * alpha * [exp(-2*alpha*(r - r0)) - exp(-alpha*(r - r0))]

        At short range (r < r0): the exp(-2*alpha*x) term dominates (x < 0, so
        both exponentials > 1, but the first more so), giving positive f (repulsive).
        At long range (r > r0): both exponentials decay, f -> 0 from below (attractive).
        """
        D_e = nn.functional.softplus(self.raw_D_e)
        alpha = nn.functional.softplus(self.raw_alpha)
        r0 = self.raw_r0

        pair_D_e = D_e[pair_indices]
        pair_alpha = alpha[pair_indices]
        pair_r0 = r0[pair_indices]

        displacement = batch.distances - pair_r0
        exp_neg_alpha_x = torch.exp(-pair_alpha * displacement)
        exp_neg_2alpha_x = exp_neg_alpha_x * exp_neg_alpha_x

        return 2.0 * pair_D_e * pair_alpha * (exp_neg_2alpha_x - exp_neg_alpha_x)

    @staticmethod
    def equilibrium_distance_param():
        return 'r0'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Morse: r0 = peak directly (minimum of V(r) is at r = r0)."""
        return peak_distance

    def lammps_pair_style(self, cutoff_angstrom):
        return f"morse {cutoff_angstrom}"

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        params = self.get_params_display()
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            d_e = params['D_e'][i]
            alpha = params['alpha'][i]
            r0 = params['r0'][i]
            lines.append(
                f"pair_coeff {t1} {t2} {d_e:.8f} {alpha:.8f} {r0:.8f} {cutoff_angstrom}"
            )
        return lines

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'D_e': params['D_e'].detach().cpu().numpy() * HARTREE_TO_EV,
            'alpha': params['alpha'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            'r0': params['r0'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = D_e * (1 - exp(-alpha*(r - r0)))^2 for a single pair type."""
        D_e = nn.functional.softplus(self.raw_D_e)[pair_index]
        alpha = nn.functional.softplus(self.raw_alpha)[pair_index]
        r0 = self.raw_r0[pair_index]

        displacement = r_bohr - r0
        exp_term = torch.exp(-alpha * displacement)
        return D_e * (1.0 - exp_term) ** 2
