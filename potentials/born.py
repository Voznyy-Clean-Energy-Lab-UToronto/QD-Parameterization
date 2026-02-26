"""Born-Mayer-Huggins pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr

# Unit conversions for dispersion coefficients:
# C has units of energy * distance^6  (Hartree * Bohr^6 internally)
HARTREE_BOHR6_TO_EV_ANG6 = Hartree * Bohr ** 6
EV_ANG6_TO_HARTREE_BOHR6 = 1.0 / HARTREE_BOHR6_TO_EV_ANG6

# D has units of energy * distance^8  (Hartree * Bohr^8 internally)
HARTREE_BOHR8_TO_EV_ANG8 = Hartree * Bohr ** 8
EV_ANG8_TO_HARTREE_BOHR8 = 1.0 / HARTREE_BOHR8_TO_EV_ANG8


class BornMayerHuggins(BasePotential):
    """Born-Mayer-Huggins pair potential.

    V(r) = A * exp(B * (sigma - r)) - C / r^6 - D / r^8

    The first term is a short-range exponential repulsion parameterized by a
    prefactor A, a stiffness B, and a size parameter sigma. The second and
    third terms are attractive dispersion (van der Waals) contributions:
    dipole-dipole (C/r^6) and dipole-quadrupole (D/r^8).

    This functional form is widely used for ionic systems (alkali halides,
    metal oxides) where the Born-Mayer exponential repulsion captures the
    Pauli repulsion between closed-shell ions.

    Parameters (internal, atomic units):
        A:     repulsive prefactor (Hartree), constrained positive via softplus
        B:     repulsive stiffness (1/Bohr), constrained positive via softplus
        sigma: ionic size parameter (Bohr), unconstrained
        C:     dipole-dipole dispersion (Hartree * Bohr^6), constrained positive via softplus
        D:     dipole-quadrupole dispersion (Hartree * Bohr^8), constrained positive via softplus

    The force is:
        f(r) = -dV/dr = A * B * exp(B * (sigma - r)) - 6*C / r^7 - 8*D / r^9

    LAMMPS convention matches: pair_style born uses V = A*exp(B*(sigma-r)) - C/r^6 - D/r^8
    with pair_coeff format: pair_coeff type1 type2 A B sigma C D
    """

    @staticmethod
    def user_param_names():
        return ['A', 'B', 'sigma', 'C', 'D']

    @staticmethod
    def param_kinds():
        return {
            'A': 'energy', 'B': 'inv_distance', 'sigma': 'distance',
            'C': 'energy_dist6', 'D': 'energy_dist8',
        }

    @staticmethod
    def parameter_names():
        return ['raw_A', 'raw_B', 'raw_sigma', 'raw_C', 'raw_D']

    def create_parameters(self, num_pairs, initial_values):
        # A: default 1.0 eV, converted to Hartree, softplus-constrained
        a_init = torch.tensor(
            initial_values.get('A', np.full(num_pairs, 1.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_A = nn.Parameter(inverse_softplus(a_init))

        # B: default 1.0 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # Conversion: B[1/Bohr] = B[1/Ang] * BOHR_TO_ANGSTROM
        b_init = torch.tensor(
            initial_values.get('B', np.full(num_pairs, 1.0 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_B = nn.Parameter(inverse_softplus(b_init))

        # sigma: default 3.0 Angstrom, converted to Bohr, unconstrained
        sigma_init = torch.tensor(
            initial_values.get('sigma', np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_sigma = nn.Parameter(sigma_init)

        # C: default 1.0 eV*Ang^6, converted to Hartree*Bohr^6, softplus-constrained
        c_init = torch.tensor(
            initial_values.get('C', np.full(num_pairs, 1.0 * EV_ANG6_TO_HARTREE_BOHR6)),
            dtype=torch.float64,
        )
        self.raw_C = nn.Parameter(inverse_softplus(c_init))

        # D: default 1.0 eV*Ang^8, converted to Hartree*Bohr^8, softplus-constrained
        d_init = torch.tensor(
            initial_values.get('D', np.full(num_pairs, 1.0 * EV_ANG8_TO_HARTREE_BOHR8)),
            dtype=torch.float64,
        )
        self.raw_D = nn.Parameter(inverse_softplus(d_init))

    def get_constrained_params(self):
        return {
            'A': nn.functional.softplus(self.raw_A),
            'B': nn.functional.softplus(self.raw_B),
            'sigma': self.raw_sigma,
            'C': nn.functional.softplus(self.raw_C),
            'D': nn.functional.softplus(self.raw_D),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = A * B * exp(B * (sigma - r)) - 6*C / r^7 - 8*D / r^9

        The exponential term is always positive (repulsive) and dominates at
        short range. The -C/r^7 and -D/r^9 terms are negative (attractive
        dispersion, pulling the dispersive terms back in). At equilibrium these
        balance; beyond that the interaction is weakly attractive and decays as
        a power law.
        """
        A = nn.functional.softplus(self.raw_A)
        B = nn.functional.softplus(self.raw_B)
        sigma = self.raw_sigma
        C = nn.functional.softplus(self.raw_C)
        D = nn.functional.softplus(self.raw_D)

        pair_A = A[pair_indices]
        pair_B = B[pair_indices]
        pair_sigma = sigma[pair_indices]
        pair_C = C[pair_indices]
        pair_D = D[pair_indices]

        distances = batch.distances
        inverse_distances = batch.inverse_distances

        # Exponential repulsion term: A * B * exp(B * (sigma - r))
        exp_term = torch.exp(pair_B * (pair_sigma - distances))
        repulsive_force = pair_A * pair_B * exp_term

        # Dispersion force terms: -6*C/r^7 - 8*D/r^9
        # These are negative (attractive), so they reduce the total force
        inverse_distances_6 = inverse_distances ** 6
        inverse_distances_7 = inverse_distances * inverse_distances_6
        inverse_distances_9 = inverse_distances_7 * inverse_distances * inverse_distances

        dispersion_c_force = 6.0 * pair_C * inverse_distances_7
        dispersion_d_force = 8.0 * pair_D * inverse_distances_9

        return repulsive_force - dispersion_c_force - dispersion_d_force

    @staticmethod
    def equilibrium_distance_param():
        return 'sigma'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Born-Mayer-Huggins: sigma maps approximately to the RDF peak.

        The equilibrium distance depends on all five parameters, but sigma
        represents the sum of ionic radii and is typically close to the
        nearest-neighbor distance. The peak is used directly.
        """
        return peak_distance

    def lammps_pair_style(self, cutoff_angstrom):
        return f"born {cutoff_angstrom}"

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        """Generate LAMMPS pair_coeff lines.

        LAMMPS born format: pair_coeff type1 type2 A B sigma C D
        where A is in eV, B in 1/Angstrom, sigma in Angstrom,
        C in eV*Ang^6, D in eV*Ang^8.
        """
        params = self.get_params_display()
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            a = params['A'][i]
            b = params['B'][i]
            sig = params['sigma'][i]
            c = params['C'][i]
            d = params['D'][i]
            lines.append(
                f"pair_coeff {t1} {t2} {a:.8f} {b:.8f} {sig:.8f} {c:.8f} {d:.8f}"
            )
        return lines

    def get_params_display(self):
        """Return parameters in physical units (eV, 1/Ang, Ang, eV*Ang^6, eV*Ang^8)."""
        params = self.get_constrained_params()
        return {
            # A: Hartree -> eV
            'A': params['A'].detach().cpu().numpy() * HARTREE_TO_EV,
            # B: 1/Bohr -> 1/Angstrom = divide by BOHR_TO_ANGSTROM
            'B': params['B'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            # sigma: Bohr -> Angstrom
            'sigma': params['sigma'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            # C: Hartree*Bohr^6 -> eV*Ang^6
            'C': params['C'].detach().cpu().numpy() * HARTREE_BOHR6_TO_EV_ANG6,
            # D: Hartree*Bohr^8 -> eV*Ang^8
            'D': params['D'].detach().cpu().numpy() * HARTREE_BOHR8_TO_EV_ANG8,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = A*exp(B*(sigma-r)) - C/r^6 - D/r^8 for a single pair type."""
        A = nn.functional.softplus(self.raw_A)[pair_index]
        B = nn.functional.softplus(self.raw_B)[pair_index]
        sigma = self.raw_sigma[pair_index]
        C = nn.functional.softplus(self.raw_C)[pair_index]
        D = nn.functional.softplus(self.raw_D)[pair_index]

        exp_term = A * torch.exp(B * (sigma - r_bohr))
        dispersion_c = C / r_bohr ** 6
        dispersion_d = D / r_bohr ** 8
        return exp_term - dispersion_c - dispersion_d
