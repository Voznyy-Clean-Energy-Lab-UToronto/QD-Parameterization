"""Yukawa (screened Coulomb) pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr

# Unit conversion for the A parameter:
# Internal: Hartree * Bohr  (energy * distance)
# Display:  eV * Angstrom
HARTREE_BOHR_TO_EV_ANG = Hartree * Bohr
EV_ANG_TO_HARTREE_BOHR = 1.0 / HARTREE_BOHR_TO_EV_ANG


class Yukawa(BasePotential):
    """Yukawa (screened Coulomb) pair potential.

    V(r) = A * exp(-kappa * r) / r
    f(r) = -dV/dr = A * exp(-kappa * r) * (kappa * r + 1) / r^2

    This describes a screened Coulomb interaction where kappa is the inverse
    screening length. At short range it behaves like A/r (bare Coulomb);
    at long range the exponential screening kills the interaction.

    Parameters (internal, atomic units):
        A:     coupling strength (Hartree * Bohr), unconstrained (can be + or -)
        kappa: inverse screening length (1/Bohr), constrained positive via softplus

    Note: A > 0 gives repulsive interaction; A < 0 gives attractive.
    A is unconstrained to allow both signs.

    LAMMPS export: Since LAMMPS pair_style yukawa only supports a global kappa,
    per-pair kappa requires tabulated export. This potential returns None for
    lammps_pair_style and generates table files via generate_table_file().
    """

    @staticmethod
    def user_param_names():
        return ['A', 'kappa']

    @staticmethod
    def param_kinds():
        return {'A': 'energy_dist', 'kappa': 'inv_distance'}

    @staticmethod
    def parameter_names():
        return ['raw_A', 'raw_kappa']

    def create_parameters(self, num_pairs, initial_values):
        # A: default 1.0 eV*Ang, converted to Hartree*Bohr, unconstrained
        a_init = torch.tensor(
            initial_values.get('A', np.full(num_pairs, 1.0 * EV_ANG_TO_HARTREE_BOHR)),
            dtype=torch.float64,
        )
        self.raw_A = nn.Parameter(a_init)

        # kappa: default 1.0 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # Conversion: kappa[1/Bohr] = kappa[1/Ang] * BOHR_TO_ANGSTROM
        kappa_init = torch.tensor(
            initial_values.get('kappa', np.full(num_pairs, 1.0 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_kappa = nn.Parameter(inverse_softplus(kappa_init))

    def get_constrained_params(self):
        return {
            'A': self.raw_A,
            'kappa': nn.functional.softplus(self.raw_kappa),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = A * exp(-kappa * r) * (kappa * r + 1) / r^2

        For A > 0 (repulsive): f is positive everywhere (exponentially screened
        repulsion). For A < 0 (attractive): f is negative (screened attraction).
        """
        A = self.raw_A
        kappa = nn.functional.softplus(self.raw_kappa)

        pair_A = A[pair_indices]
        pair_kappa = kappa[pair_indices]

        distances = batch.distances
        inverse_distances = batch.inverse_distances

        exp_term = torch.exp(-pair_kappa * distances)
        kappa_r_plus_one = pair_kappa * distances + 1.0

        # f(r) = A * exp(-kappa*r) * (kappa*r + 1) / r^2
        return pair_A * exp_term * kappa_r_plus_one * inverse_distances * inverse_distances

    @staticmethod
    def equilibrium_distance_param():
        return 'kappa'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For Yukawa: map RDF peak to an estimate of the inverse screening length.

        The Yukawa potential has no true equilibrium distance (it is purely
        repulsive for A > 0 or purely attractive for A < 0). As a rough
        initialization, we set kappa = 1 / peak_distance, placing the screening
        length at the characteristic interaction distance.
        """
        return 1.0 / peak_distance

    def lammps_pair_style(self, cutoff_angstrom):
        """Return None: LAMMPS yukawa only supports a global kappa.

        Per-pair kappa values require tabulated export via generate_table_file().
        """
        return None

    def lammps_pair_coeff_lines(self, pair_names, elements, cutoff_angstrom):
        """Generate LAMMPS pair_coeff lines for table-based export.

        Since LAMMPS pair_style yukawa only supports a single global kappa,
        per-pair parameters must be exported as pair_style table files.
        """
        elem_to_type = {e: i + 1 for i, e in enumerate(elements)}
        lines = []
        for i, pair_name in enumerate(pair_names):
            e1, e2 = pair_name.split('-')
            t1, t2 = elem_to_type[e1], elem_to_type[e2]
            table_file = f"yukawa_{pair_name}.table"
            keyword = pair_name.replace('-', '_')
            lines.append(
                f"pair_coeff {t1} {t2} {table_file} {keyword}"
            )
        return lines

    def get_params_display(self):
        """Return parameters in physical units (eV*Angstrom, 1/Angstrom)."""
        params = self.get_constrained_params()
        return {
            # A: Hartree*Bohr -> eV*Angstrom
            'A': params['A'].detach().cpu().numpy() * HARTREE_BOHR_TO_EV_ANG,
            # kappa: 1/Bohr -> 1/Angstrom = divide by BOHR_TO_ANGSTROM
            'kappa': params['kappa'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = A * exp(-kappa * r) / r for a single pair type."""
        A = self.raw_A[pair_index]
        kappa = nn.functional.softplus(self.raw_kappa)[pair_index]

        return A * torch.exp(-kappa * r_bohr) / r_bohr
