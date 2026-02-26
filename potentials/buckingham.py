"""Buckingham pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr

# Unit conversion for the C parameter:
# Internal: Hartree * Bohr^6
# Display:  eV * Angstrom^6
# C_display = C_internal * (Hartree / 1) * (Bohr / 1)^6
HARTREE_BOHR6_TO_EV_ANG6 = Hartree * Bohr ** 6
EV_ANG6_TO_HARTREE_BOHR6 = 1.0 / HARTREE_BOHR6_TO_EV_ANG6


class Buckingham(BasePotential):
    """Buckingham (exp-6) pair potential.

    V(r) = A * exp(-r / rho) - C / r^6
    f(r) = -dV/dr = (A / rho) * exp(-r / rho) - 6 * C / r^7

    Parameters (internal, atomic units):
        A:   prefactor (Hartree), constrained positive via softplus
        rho: length scale (Bohr), constrained positive via softplus
        C:   dispersion coefficient (Hartree * Bohr^6), constrained positive via softplus

    Note: The Buckingham potential has an unphysical attraction at very short range
    (the -C/r^6 term diverges faster than the exponential repulsion). This is handled
    by the cutoff/smoothing and should not cause issues at physical distances.
    """

    @staticmethod
    def user_param_names():
        return ['A', 'rho', 'C']

    @staticmethod
    def param_kinds():
        return {'A': 'energy', 'rho': 'distance', 'C': 'energy_dist6'}

    @staticmethod
    def parameter_names():
        return ['raw_A', 'raw_rho', 'raw_C']

    def create_parameters(self, num_pairs, initial_values):
        # A: default 1000 eV, converted to Hartree, softplus-constrained
        a_init = torch.tensor(
            initial_values.get('A', np.full(num_pairs, 1000.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_A = nn.Parameter(inverse_softplus(a_init))

        # rho: default 0.3 Angstrom, converted to Bohr, softplus-constrained
        rho_init = torch.tensor(
            initial_values.get('rho', np.full(num_pairs, 0.3 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_rho = nn.Parameter(inverse_softplus(rho_init))

        # C: default 1.0 eV*Ang^6, converted to Hartree*Bohr^6, softplus-constrained
        c_init = torch.tensor(
            initial_values.get('C', np.full(num_pairs, 1.0 * EV_ANG6_TO_HARTREE_BOHR6)),
            dtype=torch.float64,
        )
        self.raw_C = nn.Parameter(inverse_softplus(c_init))

    def get_constrained_params(self):
        return {
            'A': nn.functional.softplus(self.raw_A),
            'rho': nn.functional.softplus(self.raw_rho),
            'C': nn.functional.softplus(self.raw_C),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = (A / rho) * exp(-r / rho) - 6 * C / r^7

        The first term (exponential repulsion) is always positive and dominates at
        short range. The second term (-6C/r^7) is negative (attractive, pulling the
        -C/r^6 dispersion back in). At very short range the exponential wins
        (repulsive); at intermediate range the balance determines the equilibrium.
        """
        A = nn.functional.softplus(self.raw_A)
        rho = nn.functional.softplus(self.raw_rho)
        C = nn.functional.softplus(self.raw_C)

        pair_A = A[pair_indices]
        pair_rho = rho[pair_indices]
        pair_C = C[pair_indices]

        distances = batch.distances
        inverse_distances = batch.inverse_distances

        exp_term = torch.exp(-distances / pair_rho)

        # 1/r^7 = (1/r) * (1/r^6) = (1/r) * (1/r)^6
        inverse_distances_6 = inverse_distances ** 6
        inverse_distances_7 = inverse_distances * inverse_distances_6

        repulsive_force = (pair_A / pair_rho) * exp_term
        attractive_force = 6.0 * pair_C * inverse_distances_7

        return repulsive_force - attractive_force

    @staticmethod
    def equilibrium_distance_param():
        return 'rho'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Buckingham: rho is a length scale, not the equilibrium distance.
        The actual equilibrium depends on all three parameters. As a rough
        initialization, we set rho to a fraction of the peak distance. The
        equilibrium for typical Buckingham parameters is at r ~ 3-4 * rho.
        """
        return peak_distance / 3.5

    def lammps_pair_style(self, cutoff_angstrom):
        return f"buck {cutoff_angstrom}"

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        params = self.get_params_display()
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            a = params['A'][i]
            rho = params['rho'][i]
            c = params['C'][i]
            # LAMMPS buck format: pair_coeff type1 type2 A rho C
            lines.append(f"pair_coeff {t1} {t2} {a:.8f} {rho:.8f} {c:.8f}")
        return lines

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'A': params['A'].detach().cpu().numpy() * HARTREE_TO_EV,
            'rho': params['rho'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'C': params['C'].detach().cpu().numpy() * HARTREE_BOHR6_TO_EV_ANG6,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = A * exp(-r/rho) - C/r^6 for a single pair type."""
        A = nn.functional.softplus(self.raw_A)[pair_index]
        rho = nn.functional.softplus(self.raw_rho)[pair_index]
        C = nn.functional.softplus(self.raw_C)[pair_index]

        exp_term = A * torch.exp(-r_bohr / rho)
        dispersion = C / r_bohr ** 6
        return exp_term - dispersion
