"""Rydberg pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr


class Rydberg(BasePotential):
    """Rydberg pair potential.

    V(r) = -D_e * (1 + alpha * (r - r_e)) * exp(-alpha * (r - r_e))

    The potential has a minimum V(r_e) = -D_e at the equilibrium distance r_e.
    alpha controls the width/stiffness of the well.

    f(r) = -dV/dr = -D_e * alpha^2 * (r - r_e) * exp(-alpha * (r - r_e))

    At short range (r < r_e): f > 0 (repulsive).
    At long range (r > r_e): f < 0 (attractive).

    Parameters (internal, atomic units):
        D_e:    well depth (Hartree), constrained positive via softplus
        alpha:  width parameter (1/Bohr), constrained positive via softplus
        r_e:    equilibrium distance (Bohr), unconstrained
    """

    @staticmethod
    def user_param_names():
        return ['D_e', 'alpha', 'r_e']

    @staticmethod
    def param_kinds():
        return {'D_e': 'energy', 'alpha': 'inv_distance', 'r_e': 'distance'}

    @staticmethod
    def parameter_names():
        return ['raw_D_e', 'raw_alpha', 'raw_r_e']

    def create_parameters(self, num_pairs, initial_values):
        # D_e: default 0.1 eV, converted to Hartree, softplus-constrained
        d_e_init = torch.tensor(
            initial_values.get('D_e', np.full(num_pairs, 0.1 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_D_e = nn.Parameter(inverse_softplus(d_e_init))

        # alpha: default 1.5 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # 1/Ang -> 1/Bohr: multiply by BOHR_TO_ANGSTROM
        alpha_init = torch.tensor(
            initial_values.get('alpha', np.full(num_pairs, 1.5 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_alpha = nn.Parameter(inverse_softplus(alpha_init))

        # r_e: default 3.0 Angstrom, converted to Bohr, unconstrained
        r_e_init = torch.tensor(
            initial_values.get('r_e', np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_r_e = nn.Parameter(r_e_init)

    def get_constrained_params(self):
        return {
            'D_e': nn.functional.softplus(self.raw_D_e),
            'alpha': nn.functional.softplus(self.raw_alpha),
            'r_e': self.raw_r_e,
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = -D_e * alpha^2 * (r - r_e) * exp(-alpha * (r - r_e))

        Derivation:
            Let u = r - r_e, so V = -D_e * (1 + alpha*u) * exp(-alpha*u).
            dV/du = -D_e * [alpha * exp(-a*u) + (1 + a*u) * (-a) * exp(-a*u)]
                  = -D_e * exp(-a*u) * [a - a - a^2*u]
                  = D_e * a^2 * u * exp(-a*u)
            f = -dV/dr = -D_e * a^2 * u * exp(-a*u)
        """
        D_e = nn.functional.softplus(self.raw_D_e)
        alpha = nn.functional.softplus(self.raw_alpha)
        r_e = self.raw_r_e

        pair_D_e = D_e[pair_indices]
        pair_alpha = alpha[pair_indices]
        pair_r_e = r_e[pair_indices]

        displacement = batch.distances - pair_r_e
        exp_term = torch.exp(-pair_alpha * displacement)

        return -pair_D_e * pair_alpha ** 2 * displacement * exp_term

    @staticmethod
    def equilibrium_distance_param():
        return 'r_e'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Rydberg: r_e = peak directly (minimum of V(r) is at r = r_e)."""
        return peak_distance

    def lammps_pair_style(self, cutoff_angstrom):
        """Rydberg has no native LAMMPS pair_style; use tabulated."""
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
            'D_e': params['D_e'].detach().cpu().numpy() * HARTREE_TO_EV,
            'alpha': params['alpha'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            'r_e': params['r_e'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = -D_e * (1 + alpha*(r - r_e)) * exp(-alpha*(r - r_e))."""
        D_e = nn.functional.softplus(self.raw_D_e)[pair_index]
        alpha = nn.functional.softplus(self.raw_alpha)[pair_index]
        r_e = self.raw_r_e[pair_index]

        displacement = r_bohr - r_e
        return -D_e * (1.0 + alpha * displacement) * torch.exp(-alpha * displacement)
