"""Tersoff many-body potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .many_body_base import ManyBodyPotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr


class Tersoff(ManyBodyPotential):
    """Tersoff many-body potential.

    V_ij = f_c(r_ij) * [f_R(r_ij) + b_ij * f_A(r_ij)]

    where:
        f_R(r) = A * exp(-lambda1 * r)           (repulsive)
        f_A(r) = -B * exp(-lambda2 * r)           (attractive)
        f_c(r) = cosine taper: 1 -> 0 over [R-D, R+D]

        b_ij = (1 + beta^n * zeta_ij^n)^(-1/(2n))
        zeta_ij = sum_k f_c(r_ik) * g(theta_ijk)
        g(theta) = 1 + c^2/d^2 - c^2/(d^2 + (h - cos(theta))^2)

    11 parameters per pair: A, B, lambda1, lambda2, beta, n, c, d, h, R, D

    Two-body params (A, B, lambda1, lambda2, beta, n, R, D) use the i-j pair type.
    Angular params (c, d, h) use the i-k pair type for triplet (i, j, k).
    """

    @staticmethod
    def user_param_names():
        return ['A', 'B', 'lambda1', 'lambda2', 'beta', 'n',
                'c', 'd', 'h', 'R', 'D']

    @staticmethod
    def param_kinds():
        return {
            'A': 'energy',
            'B': 'energy',
            'lambda1': 'inv_distance',
            'lambda2': 'inv_distance',
            'beta': 'dimensionless',
            'n': 'dimensionless',
            'c': 'dimensionless',
            'd': 'dimensionless',
            'h': 'dimensionless',
            'R': 'distance',
            'D': 'distance',
        }

    @staticmethod
    def parameter_names():
        return ['raw_A', 'raw_B', 'raw_lambda1', 'raw_lambda2',
                'raw_beta', 'raw_n', 'raw_c', 'raw_d', 'raw_h',
                'raw_R', 'raw_D']

    def create_parameters(self, num_pairs, initial_values):
        # A: repulsive prefactor (Hartree), softplus
        A_init = torch.tensor(
            initial_values.get('A', np.full(num_pairs, 3000.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_A = nn.Parameter(inverse_softplus(A_init))

        # B: attractive prefactor (Hartree), softplus
        B_init = torch.tensor(
            initial_values.get('B', np.full(num_pairs, 300.0 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_B = nn.Parameter(inverse_softplus(B_init))

        # lambda1: 1/Bohr, softplus
        # 1/Ang -> 1/Bohr: multiply by BOHR_TO_ANGSTROM
        lambda1_init = torch.tensor(
            initial_values.get('lambda1', np.full(num_pairs, 2.5 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_lambda1 = nn.Parameter(inverse_softplus(lambda1_init))

        # lambda2: 1/Bohr, softplus
        lambda2_init = torch.tensor(
            initial_values.get('lambda2', np.full(num_pairs, 1.8 * BOHR_TO_ANGSTROM)),
            dtype=torch.float64,
        )
        self.raw_lambda2 = nn.Parameter(inverse_softplus(lambda2_init))

        # beta: dimensionless, softplus
        beta_init = torch.tensor(
            initial_values.get('beta', np.full(num_pairs, 1e-6)),
            dtype=torch.float64,
        )
        self.raw_beta = nn.Parameter(inverse_softplus(beta_init))

        # n: dimensionless, softplus
        n_init = torch.tensor(
            initial_values.get('n', np.full(num_pairs, 1.0)),
            dtype=torch.float64,
        )
        self.raw_n = nn.Parameter(inverse_softplus(n_init))

        # c: dimensionless, softplus
        c_init = torch.tensor(
            initial_values.get('c', np.full(num_pairs, 100.0)),
            dtype=torch.float64,
        )
        self.raw_c = nn.Parameter(inverse_softplus(c_init))

        # d: dimensionless, softplus
        d_init = torch.tensor(
            initial_values.get('d', np.full(num_pairs, 10.0)),
            dtype=torch.float64,
        )
        self.raw_d = nn.Parameter(inverse_softplus(d_init))

        # h: dimensionless, tanh -> [-1, 1]
        # inverse tanh: atanh(x)
        h_init = torch.tensor(
            initial_values.get('h', np.full(num_pairs, 0.0)),
            dtype=torch.float64,
        )
        self.raw_h = nn.Parameter(torch.atanh(h_init.clamp(-0.999, 0.999)))

        # R: cutoff center (Bohr), softplus
        R_init = torch.tensor(
            initial_values.get('R', np.full(num_pairs, 3.5 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_R = nn.Parameter(inverse_softplus(R_init))

        # D: cutoff width (Bohr), softplus
        D_init = torch.tensor(
            initial_values.get('D', np.full(num_pairs, 0.2 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_D = nn.Parameter(inverse_softplus(D_init))

    def get_constrained_params(self):
        return {
            'A': nn.functional.softplus(self.raw_A),
            'B': nn.functional.softplus(self.raw_B),
            'lambda1': nn.functional.softplus(self.raw_lambda1),
            'lambda2': nn.functional.softplus(self.raw_lambda2),
            'beta': nn.functional.softplus(self.raw_beta),
            'n': nn.functional.softplus(self.raw_n),
            'c': nn.functional.softplus(self.raw_c),
            'd': nn.functional.softplus(self.raw_d),
            'h': torch.tanh(self.raw_h),
            'R': nn.functional.softplus(self.raw_R),
            'D': nn.functional.softplus(self.raw_D),
        }

    def compute_energy(self, positions, edge_index, element_indices,
                       pair_lookup_table, batch_vector, num_atoms):
        """Compute total Tersoff energy."""
        params = self.get_constrained_params()
        source, target = edge_index

        # Compute distances from positions (for autograd)
        disp = positions[target] - positions[source]
        dist = torch.norm(disp, dim=1)

        # Pair types for edges
        src_elem = element_indices[source]
        tgt_elem = element_indices[target]
        pair_idx = pair_lookup_table[src_elem, tgt_elem]

        # Gather two-body params per edge
        A = params['A'][pair_idx]
        B = params['B'][pair_idx]
        lam1 = params['lambda1'][pair_idx]
        lam2 = params['lambda2'][pair_idx]
        beta = params['beta'][pair_idx]
        n = params['n'][pair_idx]
        R = params['R'][pair_idx]
        D = params['D'][pair_idx]

        # Cutoff function per edge
        fc = self._cutoff(dist, R, D)

        # Repulsive and attractive two-body terms per edge
        f_R = A * torch.exp(-lam1 * dist)
        f_A = -B * torch.exp(-lam2 * dist)

        # Build triplets for angular terms
        triplet_i, triplet_j, triplet_k, eidx_ij, eidx_ik = \
            self.build_triplets(edge_index, num_atoms)

        if triplet_i.shape[0] > 0:
            # Distances for ij and ik edges
            r_ij = dist[eidx_ij]
            r_ik = dist[eidx_ik]

            # Cutoff on r_ik
            R_ik = params['R'][pair_lookup_table[
                element_indices[triplet_i], element_indices[triplet_k]]]
            D_ik = params['D'][pair_lookup_table[
                element_indices[triplet_i], element_indices[triplet_k]]]
            fc_ik = self._cutoff(r_ik, R_ik, D_ik)

            # Cos(theta_ijk) from positions
            disp_ij = disp[eidx_ij]  # positions[j] - positions[i]
            disp_ik = disp[eidx_ik]  # positions[k] - positions[i]
            cos_theta = (disp_ij * disp_ik).sum(dim=1) / (r_ij * r_ik + 1e-20)

            # Angular params use i-k pair type
            ik_pair = pair_lookup_table[
                element_indices[triplet_i], element_indices[triplet_k]]
            c = params['c'][ik_pair]
            d = params['d'][ik_pair]
            h = params['h'][ik_pair]

            # g(theta) = 1 + c^2/d^2 - c^2/(d^2 + (h - cos_theta)^2)
            c_sq = c * c
            d_sq = d * d
            h_cos = h - cos_theta
            g = 1.0 + c_sq / d_sq - c_sq / (d_sq + h_cos * h_cos)

            # zeta_ij = sum_k fc(r_ik) * g(theta_ijk)
            zeta_contrib = fc_ik * g
            # Scatter-add zeta contributions to each ij edge
            num_edges = dist.shape[0]
            zeta = torch.zeros(num_edges, dtype=positions.dtype,
                               device=positions.device)
            zeta.scatter_add_(0, eidx_ij, zeta_contrib)
        else:
            zeta = torch.zeros_like(dist)

        # Bond order: b_ij = (1 + beta^n * zeta^n)^(-1/(2n))
        # Use safe power with clamping to avoid NaN gradients
        n_safe = n[pair_idx] if triplet_i.shape[0] > 0 else n[pair_idx]
        # Gather per-edge beta and n
        beta_e = beta
        n_e = n
        # Note: beta and n were already gathered per edge above
        # Recompute for per-edge pair_idx
        beta_n = torch.pow(beta_e + 1e-30, n_e)
        zeta_n = torch.pow(zeta.clamp(min=0) + 1e-30, n_e)
        b = torch.pow(1.0 + beta_n * zeta_n, -1.0 / (2.0 * n_e))

        # Total energy: sum over edges with 0.5 factor for double-counting
        V_per_edge = fc * (f_R + b * f_A)
        V_total = 0.5 * V_per_edge.sum()

        return V_total

    @staticmethod
    def _cutoff(r, R, D):
        """Cosine taper cutoff: 1 for r < R-D, 0 for r > R+D.

        f_c(r) = 1                                        if r < R - D
               = 0.5 - 0.5 * sin(pi/2 * (r - R) / D)    if |r - R| <= D
               = 0                                        if r > R + D
        """
        inner = R - D
        outer = R + D
        fc = torch.where(
            r < inner,
            torch.ones_like(r),
            torch.where(
                r > outer,
                torch.zeros_like(r),
                0.5 - 0.5 * torch.sin(torch.pi / 2.0 * (r - R) / (D + 1e-20))
            )
        )
        return fc

    # ── Display ───────────────────────────────────────────────────────

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'A': params['A'].detach().cpu().numpy() * HARTREE_TO_EV,
            'B': params['B'].detach().cpu().numpy() * HARTREE_TO_EV,
            'lambda1': params['lambda1'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            'lambda2': params['lambda2'].detach().cpu().numpy() / BOHR_TO_ANGSTROM,
            'beta': params['beta'].detach().cpu().numpy(),
            'n': params['n'].detach().cpu().numpy(),
            'c': params['c'].detach().cpu().numpy(),
            'd': params['d'].detach().cpu().numpy(),
            'h': params['h'].detach().cpu().numpy(),
            'R': params['R'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'D': params['D'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
        }

    # ── LAMMPS export ─────────────────────────────────────────────────

    def write_lammps_file(self, pair_names, elements, cutoff_angstrom, filepath):
        """Write LAMMPS .tersoff parameter file.

        The LAMMPS tersoff format has 14 columns per line:
        element_i element_j element_k m gamma lambda3 c d costheta0 n beta
        lambda2 B R D lambda1 A

        For our implementation:
        - m = 3 (standard Tersoff)
        - gamma = 1.0 (absorbed into beta)
        - lambda3 = 0.0 (no three-body distance dependence)
        """
        params = self.get_params_display()
        tersoff_file = filepath + '.tersoff'

        with open(tersoff_file, 'w') as f:
            f.write("# Tersoff parameters fitted by ML_fitter\n")
            f.write("# Format: e1 e2 e3 m gamma lambda3 c d costheta0 "
                    "n beta lambda2 B R D lambda1 A\n\n")

            # For each element triple (i, j, k)
            for ei in elements:
                for ej in elements:
                    for ek in elements:
                        # Two-body params from i-j pair
                        ij_pair = self._canonical_pair_name(ei, ej, elements)
                        ij_idx = pair_names.index(ij_pair)

                        # Angular params from i-k pair
                        ik_pair = self._canonical_pair_name(ei, ek, elements)
                        ik_idx = pair_names.index(ik_pair)

                        A_val = params['A'][ij_idx]
                        B_val = params['B'][ij_idx]
                        lam1 = params['lambda1'][ij_idx]
                        lam2 = params['lambda2'][ij_idx]
                        beta_val = params['beta'][ij_idx]
                        n_val = params['n'][ij_idx]
                        R_val = params['R'][ij_idx]
                        D_val = params['D'][ij_idx]

                        c_val = params['c'][ik_idx]
                        d_val = params['d'][ik_idx]
                        h_val = params['h'][ik_idx]

                        # LAMMPS format: m gamma lambda3 c d costheta0 n beta
                        # lambda2 B R D lambda1 A
                        f.write(
                            f"{ei:>4s} {ej:>4s} {ek:>4s}  "
                            f"3  1.0  0.0  "
                            f"{c_val:.8e}  {d_val:.8e}  {h_val:.8e}  "
                            f"{n_val:.8e}  {beta_val:.8e}  "
                            f"{lam2:.8e}  {B_val:.8e}  "
                            f"{R_val:.8e}  {D_val:.8e}  "
                            f"{lam1:.8e}  {A_val:.8e}\n"
                        )

        pair_style = f"tersoff"
        pair_coeff = [f"pair_coeff * * {tersoff_file} " +
                      " ".join(elements)]
        return pair_style, pair_coeff, [tersoff_file]

    @staticmethod
    def _canonical_pair_name(e1, e2, elements):
        """Get canonical pair name (sorted by element order)."""
        i1 = elements.index(e1)
        i2 = elements.index(e2)
        if i1 <= i2:
            return f"{e1}-{e2}"
        return f"{e2}-{e1}"
