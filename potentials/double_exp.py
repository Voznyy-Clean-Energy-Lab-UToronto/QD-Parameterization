"""Double-exponential pair potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .base import BasePotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr


class DoubleExponential(BasePotential):
    """Double-exponential (exp-exp) pair potential.

    V(r) = A_rep * exp(-B_rep * r) - C_att * exp(-D_att * r)

    A purely exponential form with separate repulsive and attractive terms.
    The repulsive term (A_rep, B_rep) should have a larger prefactor and
    steeper decay; the attractive term (C_att, D_att) has a smaller prefactor
    and slower decay, creating a bound well at intermediate distances.

    f(r) = -dV/dr = A_rep * B_rep * exp(-B_rep * r)
                   - C_att * D_att * exp(-D_att * r)

    Parameters (internal, atomic units):
        A_rep:  repulsive prefactor (Hartree), constrained positive via softplus
        B_rep:  repulsive decay rate (1/Bohr), constrained positive via softplus
        C_att:  attractive prefactor (Hartree), constrained positive via softplus
        D_att:  attractive decay rate (1/Bohr), constrained positive via softplus
    """

    @staticmethod
    def user_param_names():
        return ['A_rep', 'B_rep', 'C_att', 'D_att']

    @staticmethod
    def param_kinds():
        return {
            'A_rep': 'energy', 'B_rep': 'inv_distance',
            'C_att': 'energy', 'D_att': 'inv_distance',
        }

    @staticmethod
    def parameter_names():
        return ['raw_A_rep', 'raw_B_rep', 'raw_C_att', 'raw_D_att']

    def create_parameters(self, num_pairs, initial_values):
        # A_rep: default 100 eV, converted to Hartree, softplus-constrained
        a_rep_init = torch.tensor(
            initial_values.get('A_rep', np.full(num_pairs, 100.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_A_rep = nn.Parameter(inverse_softplus(a_rep_init))

        # B_rep: default 2.0 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # 1/Ang -> 1/Bohr: multiply by BOHR_TO_ANGSTROM
        b_rep_init = torch.tensor(
            initial_values.get('B_rep', np.full(num_pairs, 2.0 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_B_rep = nn.Parameter(inverse_softplus(b_rep_init))

        # C_att: default 1.0 eV, converted to Hartree, softplus-constrained
        c_att_init = torch.tensor(
            initial_values.get('C_att', np.full(num_pairs, 1.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_C_att = nn.Parameter(inverse_softplus(c_att_init))

        # D_att: default 1.0 1/Angstrom, converted to 1/Bohr, softplus-constrained
        # 1/Ang -> 1/Bohr: multiply by BOHR_TO_ANGSTROM
        d_att_init = torch.tensor(
            initial_values.get('D_att', np.full(num_pairs, 1.0 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_D_att = nn.Parameter(inverse_softplus(d_att_init))

    def get_constrained_params(self):
        return {
            'A_rep': nn.functional.softplus(self.raw_A_rep),
            'B_rep': nn.functional.softplus(self.raw_B_rep),
            'C_att': nn.functional.softplus(self.raw_C_att),
            'D_att': nn.functional.softplus(self.raw_D_att),
        }

    @staticmethod
    def required_edge_fields():
        return ['distances']

    def scalar_force(self, batch, pair_indices):
        """Compute -dV/dr for each edge.

        f(r) = A_rep * B_rep * exp(-B_rep * r) - C_att * D_att * exp(-D_att * r)

        The first term is the repulsive force contribution (always positive);
        the second is the attractive force contribution (always positive, subtracted).
        Net force is repulsive at short range where the steeper B_rep exponential
        dominates, and attractive at long range.
        """
        A_rep = nn.functional.softplus(self.raw_A_rep)
        B_rep = nn.functional.softplus(self.raw_B_rep)
        C_att = nn.functional.softplus(self.raw_C_att)
        D_att = nn.functional.softplus(self.raw_D_att)

        pair_A_rep = A_rep[pair_indices]
        pair_B_rep = B_rep[pair_indices]
        pair_C_att = C_att[pair_indices]
        pair_D_att = D_att[pair_indices]

        distances = batch.distances

        repulsive_force = pair_A_rep * pair_B_rep * torch.exp(-pair_B_rep * distances)
        attractive_force = pair_C_att * pair_D_att * torch.exp(-pair_D_att * distances)

        return repulsive_force - attractive_force

    @staticmethod
    def equilibrium_distance_param():
        return 'B_rep'

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        """For DoubleExponential: B_rep is an inverse length scale.

        A reasonable initialization is B_rep ~ 1/peak_distance, so the
        repulsive exponential has a length scale comparable to the nearest-
        neighbor distance.
        """
        return 1.0 / peak_distance

    def lammps_pair_style(self, cutoff_angstrom):
        """No native LAMMPS pair_style; use tabulated."""
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
            'A_rep': params['A_rep'].detach().cpu().numpy() * HARTREE_TO_EV,
            'B_rep': params['B_rep'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            'C_att': params['C_att'].detach().cpu().numpy() * HARTREE_TO_EV,
            'D_att': params['D_att'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
        }

    def _pair_energy(self, r_bohr, pair_index):
        """V(r) = A_rep * exp(-B_rep * r) - C_att * exp(-D_att * r)."""
        A_rep = nn.functional.softplus(self.raw_A_rep)[pair_index]
        B_rep = nn.functional.softplus(self.raw_B_rep)[pair_index]
        C_att = nn.functional.softplus(self.raw_C_att)[pair_index]
        D_att = nn.functional.softplus(self.raw_D_att)[pair_index]

        repulsive = A_rep * torch.exp(-B_rep * r_bohr)
        attractive = C_att * torch.exp(-D_att * r_bohr)
        return repulsive - attractive
