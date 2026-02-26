"""Damped Buckingham pair potential with Tang-Toennies damping."""

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
#   C6 has units of energy * distance^6
#   C8 has units of energy * distance^8
HARTREE_BOHR6_TO_EV_ANG6 = Hartree * Bohr ** 6
EV_ANG6_TO_HARTREE_BOHR6 = 1.0 / HARTREE_BOHR6_TO_EV_ANG6

HARTREE_BOHR8_TO_EV_ANG8 = Hartree * Bohr ** 8
EV_ANG8_TO_HARTREE_BOHR8 = 1.0 / HARTREE_BOHR8_TO_EV_ANG8


def _tang_toennies(n, x):
    """Tang-Toennies damping function of order n.

    f_n(x) = 1 - exp(-x) * sum_{k=0}^{n} x^k / k!

    This smoothly damps the 1/r^n dispersion at short range, preventing
    the unphysical divergence of the bare dispersion term. f_n(x) -> 0
    as x -> 0 (full damping) and f_n(x) -> 1 as x -> infinity (no damping).

    Args:
        n: damping order (integer, typically 6 or 8)
        x: tensor of b*r values (dimensionless)

    Returns:
        tensor of damping values in [0, 1]
    """
    # Compute the incomplete gamma series: sum_{k=0}^{n} x^k / k!
    series_sum = torch.zeros_like(x)
    term = torch.ones_like(x)  # x^0 / 0! = 1
    series_sum = series_sum + term
    for k in range(1, n + 1):
        term = term * x / k
        series_sum = series_sum + term

    return 1.0 - torch.exp(-x) * series_sum


def _tang_toennies_derivative(n, x):
    """Derivative of the Tang-Toennies damping function with respect to x.

    df_n/dx = exp(-x) * x^n / n!

    This is used in the analytical force computation.

    Args:
        n: damping order (integer, typically 6 or 8)
        x: tensor of b*r values (dimensionless)

    Returns:
        tensor of derivative values
    """
    # Compute x^n / n! iteratively for numerical stability
    # x^n / n! = prod_{k=1}^{n} (x/k)
    xn_over_nfact = torch.ones_like(x)
    for k in range(1, n + 1):
        xn_over_nfact = xn_over_nfact * x / k

    return torch.exp(-x) * xn_over_nfact


class DampedBuckingham(BasePotential):
    """Damped Buckingham potential with Tang-Toennies damping on dispersion.

    V(r) = A * exp(-r/rho) - f6(b*r) * C6/r^6 - f8(b*r) * C8/r^8

    where f_n(x) = 1 - exp(-x) * sum_{k=0}^{n} x^k/k! is the Tang-Toennies
    damping function. The damping prevents the unphysical short-range divergence
    of the C6/r^6 and C8/r^8 dispersion terms, which in the standard Buckingham
    potential can overwhelm the exponential repulsion at very short range.

    Force (analytical):
        f(r) = -dV/dr
             = (A/rho) * exp(-r/rho)
               - b * f6'(b*r) * C6/r^6 + 6 * f6(b*r) * C6/r^7
               - b * f8'(b*r) * C8/r^8 + 8 * f8(b*r) * C8/r^9

        where f_n'(x) = df_n/dx = exp(-x) * x^n / n!

    Parameters (internal, atomic units):
        A:   repulsive prefactor (Hartree), constrained positive via softplus
        rho: repulsive length scale (Bohr), constrained positive via softplus
        C6:  C6 dispersion coefficient (Hartree * Bohr^6), constrained positive via softplus
        C8:  C8 dispersion coefficient (Hartree * Bohr^8), constrained positive via softplus
        b:   damping range parameter (1/Bohr), constrained positive via softplus
    """

    @staticmethod
    def user_param_names():
        return ['A', 'rho', 'C6', 'C8', 'b']

    @staticmethod
    def param_kinds():
        return {
            'A': 'energy', 'rho': 'distance',
            'C6': 'energy_dist6', 'C8': 'energy_dist8', 'b': 'inv_distance',
        }

    @staticmethod
    def parameter_names():
        return ['raw_A', 'raw_rho', 'raw_C6', 'raw_C8', 'raw_b']

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

        # C6: default 1.0 eV*Ang^6, converted to Hartree*Bohr^6, softplus-constrained
        c6_init = torch.tensor(
            initial_values.get('C6', np.full(num_pairs, 1.0 * EV_ANG6_TO_HARTREE_BOHR6)),
            dtype=torch.float64,
        )
        self.raw_C6 = nn.Parameter(inverse_softplus(c6_init))

        # C8: default 0.1 eV*Ang^8, converted to Hartree*Bohr^8, softplus-constrained
        c8_init = torch.tensor(
            initial_values.get('C8', np.full(num_pairs, 0.1 * EV_ANG8_TO_HARTREE_BOHR8)),
            dtype=torch.float64,
        )
        self.raw_C8 = nn.Parameter(inverse_softplus(c8_init))

        # b: default 2.0 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # 1/Ang -> 1/Bohr: multiply by BOHR_TO_ANGSTROM
        b_init = torch.tensor(
            initial_values.get('b', np.full(num_pairs, 2.0 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_b = nn.Parameter(inverse_softplus(b_init))

    def get_constrained_params(self):
        return {
            'A': nn.functional.softplus(self.raw_A),
            'rho': nn.functional.softplus(self.raw_rho),
            'C6': nn.functional.softplus(self.raw_C6),
            'C8': nn.functional.softplus(self.raw_C8),
            'b': nn.functional.softplus(self.raw_b),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge analytically.

        f(r) = (A/rho) * exp(-r/rho)
               - b * f6'(br) * C6/r^6  +  6 * f6(br) * C6/r^7
               - b * f8'(br) * C8/r^8  +  8 * f8(br) * C8/r^9

        The repulsive Born-Mayer term contributes a positive (repulsive) force.
        Each damped dispersion term -f_n(br)*C_n/r^n contributes a force with
        two parts: one from the derivative of the damping function (negative,
        since the damping grows with r, strengthening the attraction) and one
        from the derivative of 1/r^n (positive, since the attraction weakens
        with distance).
        """
        A = nn.functional.softplus(self.raw_A)
        rho = nn.functional.softplus(self.raw_rho)
        C6 = nn.functional.softplus(self.raw_C6)
        C8 = nn.functional.softplus(self.raw_C8)
        b = nn.functional.softplus(self.raw_b)

        pair_A = A[pair_indices]
        pair_rho = rho[pair_indices]
        pair_C6 = C6[pair_indices]
        pair_C8 = C8[pair_indices]
        pair_b = b[pair_indices]

        distances = batch.distances
        inverse_distances = batch.inverse_distances
        br = pair_b * distances

        # Born-Mayer repulsive force: (A/rho) * exp(-r/rho)
        repulsive_force = (pair_A / pair_rho) * torch.exp(-distances / pair_rho)

        # Inverse distance powers needed for dispersion force terms
        inv_r6 = inverse_distances ** 6
        inv_r7 = inv_r6 * inverse_distances
        inv_r8 = inv_r7 * inverse_distances
        inv_r9 = inv_r8 * inverse_distances

        # Tang-Toennies damping values and derivatives
        f6 = _tang_toennies(6, br)
        f6_prime = _tang_toennies_derivative(6, br)
        f8 = _tang_toennies(8, br)
        f8_prime = _tang_toennies_derivative(8, br)

        # Force from -C6 * f6(br) / r^6 term:
        #   -d/dr[-C6 * f6(br) / r^6] = C6 * [b * f6'(br) / r^6 - 6 * f6(br) / r^7]
        #   Contribution to f = -dV/dr: -b*f6'*C6/r^6 + 6*f6*C6/r^7
        c6_force = -pair_b * f6_prime * pair_C6 * inv_r6 + 6.0 * f6 * pair_C6 * inv_r7

        # Force from -C8 * f8(br) / r^8 term (same structure):
        c8_force = -pair_b * f8_prime * pair_C8 * inv_r8 + 8.0 * f8 * pair_C8 * inv_r9

        return repulsive_force + c6_force + c8_force

    @staticmethod
    def equilibrium_distance_param():
        return 'rho'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For DampedBuckingham: rho is a length scale, not the equilibrium distance.

        Like standard Buckingham, the equilibrium depends on all parameters.
        Set rho to a fraction of the peak distance as a rough initialization.
        """
        return peak_distance / 3.5

    def lammps_pair_style(self, cutoff_angstrom):
        """No native LAMMPS pair_style for damped Buckingham; use tabulated."""
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
            'A': params['A'].detach().cpu().numpy() * HARTREE_TO_EV,
            'rho': params['rho'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'C6': params['C6'].detach().cpu().numpy() * HARTREE_BOHR6_TO_EV_ANG6,
            'C8': params['C8'].detach().cpu().numpy() * HARTREE_BOHR8_TO_EV_ANG8,
            'b': params['b'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = A*exp(-r/rho) - f6(br)*C6/r^6 - f8(br)*C8/r^8."""
        A = nn.functional.softplus(self.raw_A)[pair_index]
        rho = nn.functional.softplus(self.raw_rho)[pair_index]
        C6 = nn.functional.softplus(self.raw_C6)[pair_index]
        C8 = nn.functional.softplus(self.raw_C8)[pair_index]
        b = nn.functional.softplus(self.raw_b)[pair_index]

        br = b * r_bohr

        repulsive = A * torch.exp(-r_bohr / rho)
        dispersion_6 = _tang_toennies(6, br) * C6 / r_bohr ** 6
        dispersion_8 = _tang_toennies(8, br) * C8 / r_bohr ** 8

        return repulsive - dispersion_6 - dispersion_8
