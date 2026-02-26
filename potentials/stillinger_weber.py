"""Stillinger-Weber many-body potential."""

import numpy as np
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr

from .many_body_base import ManyBodyPotential, inverse_softplus

HARTREE_TO_EV = Hartree
BOHR_TO_ANGSTROM = Bohr
EV_TO_HARTREE = 1.0 / Hartree
ANGSTROM_TO_BOHR = 1.0 / Bohr


class StillingerWeber(ManyBodyPotential):
    """Stillinger-Weber potential with two-body and three-body terms.

    Two-body:
        V2(r) = A * eps * (B * (sigma/r)^p - (sigma/r)^q) * exp(sigma / (r - a*sigma))
        for r < a * sigma (else 0)

    Three-body:
        V3(i,j,k) = lam * eps * (cos(theta_jik) - cos_theta0)^2
                     * exp(gamma * sigma / (r_ij - a*sigma))
                     * exp(gamma * sigma / (r_ik - a*sigma))
        for r_ij < a*sigma AND r_ik < a*sigma (else 0)

    10 parameters per pair: epsilon, sigma, a, lam, gamma, A_sw, B_sw, p, q, cos_theta0
    """

    @staticmethod
    def user_param_names():
        return ['epsilon', 'sigma', 'a', 'lam', 'gamma',
                'A_sw', 'B_sw', 'p', 'q', 'cos_theta0']

    @staticmethod
    def param_kinds():
        return {
            'epsilon': 'energy',
            'sigma': 'distance',
            'a': 'dimensionless',
            'lam': 'dimensionless',
            'gamma': 'dimensionless',
            'A_sw': 'dimensionless',
            'B_sw': 'dimensionless',
            'p': 'dimensionless',
            'q': 'dimensionless',
            'cos_theta0': 'dimensionless',
        }

    @staticmethod
    def parameter_names():
        return ['raw_epsilon', 'raw_sigma', 'raw_a', 'raw_lam', 'raw_gamma',
                'raw_A_sw', 'raw_B_sw', 'raw_p', 'raw_q', 'raw_cos_theta0']

    def create_parameters(self, num_pairs, initial_values):
        # epsilon: energy (Hartree), softplus
        eps_init = torch.tensor(
            initial_values.get('epsilon', np.full(num_pairs, 2.17 * EV_TO_HARTREE)),
            dtype=torch.float64,
        )
        self.raw_epsilon = nn.Parameter(inverse_softplus(eps_init))

        # sigma: distance (Bohr), softplus
        sigma_init = torch.tensor(
            initial_values.get('sigma', np.full(num_pairs, 2.1 * ANGSTROM_TO_BOHR)),
            dtype=torch.float64,
        )
        self.raw_sigma = nn.Parameter(inverse_softplus(sigma_init))

        # a: dimensionless cutoff multiplier, softplus
        a_init = torch.tensor(
            initial_values.get('a', np.full(num_pairs, 1.8)),
            dtype=torch.float64,
        )
        self.raw_a = nn.Parameter(inverse_softplus(a_init))

        # lam: three-body strength, softplus
        lam_init = torch.tensor(
            initial_values.get('lam', np.full(num_pairs, 21.0)),
            dtype=torch.float64,
        )
        self.raw_lam = nn.Parameter(inverse_softplus(lam_init))

        # gamma: three-body range, softplus
        gamma_init = torch.tensor(
            initial_values.get('gamma', np.full(num_pairs, 1.2)),
            dtype=torch.float64,
        )
        self.raw_gamma = nn.Parameter(inverse_softplus(gamma_init))

        # A_sw: two-body strength, softplus
        A_init = torch.tensor(
            initial_values.get('A_sw', np.full(num_pairs, 7.05)),
            dtype=torch.float64,
        )
        self.raw_A_sw = nn.Parameter(inverse_softplus(A_init))

        # B_sw: two-body repulsion, softplus
        B_init = torch.tensor(
            initial_values.get('B_sw', np.full(num_pairs, 0.6)),
            dtype=torch.float64,
        )
        self.raw_B_sw = nn.Parameter(inverse_softplus(B_init))

        # p: repulsive exponent, softplus
        p_init = torch.tensor(
            initial_values.get('p', np.full(num_pairs, 4.0)),
            dtype=torch.float64,
        )
        self.raw_p = nn.Parameter(inverse_softplus(p_init))

        # q: attractive exponent, softplus
        q_init = torch.tensor(
            initial_values.get('q', np.full(num_pairs, 0.0)),
            dtype=torch.float64,
        )
        self.raw_q = nn.Parameter(inverse_softplus(q_init))

        # cos_theta0: equilibrium angle cosine, tanh -> [-1, 1]
        cos_init = torch.tensor(
            initial_values.get('cos_theta0', np.full(num_pairs, -1.0 / 3.0)),
            dtype=torch.float64,
        )
        self.raw_cos_theta0 = nn.Parameter(
            torch.atanh(cos_init.clamp(-0.999, 0.999))
        )

    def get_constrained_params(self):
        return {
            'epsilon': nn.functional.softplus(self.raw_epsilon),
            'sigma': nn.functional.softplus(self.raw_sigma),
            'a': nn.functional.softplus(self.raw_a),
            'lam': nn.functional.softplus(self.raw_lam),
            'gamma': nn.functional.softplus(self.raw_gamma),
            'A_sw': nn.functional.softplus(self.raw_A_sw),
            'B_sw': nn.functional.softplus(self.raw_B_sw),
            'p': nn.functional.softplus(self.raw_p),
            'q': nn.functional.softplus(self.raw_q),
            'cos_theta0': torch.tanh(self.raw_cos_theta0),
        }

    def compute_energy(self, positions, edge_index, element_indices,
                       pair_lookup_table, batch_vector, num_atoms):
        """Compute total Stillinger-Weber energy."""
        params = self.get_constrained_params()
        source, target = edge_index

        # Compute distances from positions (for autograd)
        disp = positions[target] - positions[source]
        dist = torch.norm(disp, dim=1)

        # Pair types for edges
        src_elem = element_indices[source]
        tgt_elem = element_indices[target]
        pair_idx = pair_lookup_table[src_elem, tgt_elem]

        # Gather per-edge params
        eps = params['epsilon'][pair_idx]
        sigma = params['sigma'][pair_idx]
        a = params['a'][pair_idx]
        A_sw = params['A_sw'][pair_idx]
        B_sw = params['B_sw'][pair_idx]
        p = params['p'][pair_idx]
        q = params['q'][pair_idx]

        # ── Two-body term ─────────────────────────────────────────────
        # Effective cutoff: a * sigma
        r_cut = a * sigma
        # Reduced distance
        rho = sigma / dist

        # exp(sigma / (r - a*sigma)) with smooth cutoff
        dr = dist - r_cut
        # Avoid exp overflow: mask edges beyond cutoff
        within_cut = (dr < -1e-10)
        safe_dr = torch.where(within_cut, dr, -torch.ones_like(dr))
        exp_term = torch.where(
            within_cut,
            torch.exp(sigma / safe_dr),
            torch.zeros_like(dr),
        )

        V2_per_edge = A_sw * eps * (B_sw * torch.pow(rho, p) - torch.pow(rho, q)) * exp_term
        V2 = 0.5 * V2_per_edge.sum()

        # ── Three-body term ───────────────────────────────────────────
        triplet_i, triplet_j, triplet_k, eidx_ij, eidx_ik = \
            self.build_triplets(edge_index, num_atoms)

        V3 = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

        if triplet_i.shape[0] > 0:
            r_ij = dist[eidx_ij]
            r_ik = dist[eidx_ik]

            # Params for ij and ik edges
            pij = pair_lookup_table[
                element_indices[triplet_i], element_indices[triplet_j]]
            pik = pair_lookup_table[
                element_indices[triplet_i], element_indices[triplet_k]]

            eps_ij = params['epsilon'][pij]
            sigma_ij = params['sigma'][pij]
            a_ij = params['a'][pij]
            lam_ij = params['lam'][pij]
            gamma_ij = params['gamma'][pij]

            sigma_ik = params['sigma'][pik]
            a_ik = params['a'][pik]
            gamma_ik = params['gamma'][pik]

            cos_theta0 = params['cos_theta0'][pij]

            # Cutoffs
            r_cut_ij = a_ij * sigma_ij
            r_cut_ik = a_ik * sigma_ik

            dr_ij = r_ij - r_cut_ij
            dr_ik = r_ik - r_cut_ik

            within_ij = (dr_ij < -1e-10)
            within_ik = (dr_ik < -1e-10)
            both_within = within_ij & within_ik

            # Exponential terms (safe)
            safe_dr_ij = torch.where(both_within, dr_ij, -torch.ones_like(dr_ij))
            safe_dr_ik = torch.where(both_within, dr_ik, -torch.ones_like(dr_ik))

            exp_ij = torch.where(
                both_within,
                torch.exp(gamma_ij * sigma_ij / safe_dr_ij),
                torch.zeros_like(dr_ij),
            )
            exp_ik = torch.where(
                both_within,
                torch.exp(gamma_ik * sigma_ik / safe_dr_ik),
                torch.zeros_like(dr_ik),
            )

            # cos(theta_jik)
            disp_ij = disp[eidx_ij]
            disp_ik = disp[eidx_ik]
            cos_theta = (disp_ij * disp_ik).sum(dim=1) / (r_ij * r_ik + 1e-20)

            # V3 = lam * eps * (cos_theta - cos_theta0)^2 * exp_ij * exp_ik
            angle_term = (cos_theta - cos_theta0) ** 2
            V3_per_triplet = lam_ij * eps_ij * angle_term * exp_ij * exp_ik
            V3 = V3_per_triplet.sum()

        return V2 + V3

    # ── Display ───────────────────────────────────────────────────────

    def get_params_display(self):
        params = self.get_constrained_params()
        return {
            'epsilon': params['epsilon'].detach().cpu().numpy() * HARTREE_TO_EV,
            'sigma': params['sigma'].detach().cpu().numpy() * BOHR_TO_ANGSTROM,
            'a': params['a'].detach().cpu().numpy(),
            'lam': params['lam'].detach().cpu().numpy(),
            'gamma': params['gamma'].detach().cpu().numpy(),
            'A_sw': params['A_sw'].detach().cpu().numpy(),
            'B_sw': params['B_sw'].detach().cpu().numpy(),
            'p': params['p'].detach().cpu().numpy(),
            'q': params['q'].detach().cpu().numpy(),
            'cos_theta0': params['cos_theta0'].detach().cpu().numpy(),
        }

    # ── LAMMPS export ─────────────────────────────────────────────────

    def write_lammps_file(self, pair_names, elements, cutoff_angstrom, filepath):
        """Write LAMMPS .sw parameter file.

        LAMMPS SW format per line:
        element_i element_j element_k epsilon sigma a lambda gamma
        costheta0 A B p q tol
        """
        params = self.get_params_display()
        sw_file = filepath + '.sw'

        with open(sw_file, 'w') as f:
            f.write("# Stillinger-Weber parameters fitted by ML_fitter\n")
            f.write("# Format: e1 e2 e3 epsilon sigma a lambda gamma "
                    "costheta0 A B p q tol\n\n")

            for ei in elements:
                for ej in elements:
                    for ek in elements:
                        # Two-body params from i-j pair
                        ij_pair = self._canonical_pair_name(ei, ej, elements)
                        ij_idx = pair_names.index(ij_pair)

                        # Three-body params also keyed by i-j pair
                        epsilon = params['epsilon'][ij_idx]
                        sigma = params['sigma'][ij_idx]
                        a_val = params['a'][ij_idx]
                        lam = params['lam'][ij_idx]
                        gamma = params['gamma'][ij_idx]
                        A_val = params['A_sw'][ij_idx]
                        B_val = params['B_sw'][ij_idx]
                        p_val = params['p'][ij_idx]
                        q_val = params['q'][ij_idx]
                        cos0 = params['cos_theta0'][ij_idx]

                        # For two-body-only entries (j != k in LAMMPS SW),
                        # LAMMPS uses the first matching line for two-body.
                        # Three-body lines need lambda > 0.
                        f.write(
                            f"{ei:>4s} {ej:>4s} {ek:>4s}  "
                            f"{epsilon:.8e}  {sigma:.8e}  {a_val:.8e}  "
                            f"{lam:.8e}  {gamma:.8e}  "
                            f"{cos0:.8e}  {A_val:.8e}  {B_val:.8e}  "
                            f"{p_val:.8e}  {q_val:.8e}  0.0\n"
                        )

        pair_style = f"sw"
        pair_coeff = [f"pair_coeff * * {sw_file} " + " ".join(elements)]
        return pair_style, pair_coeff, [sw_file]

    @staticmethod
    def _canonical_pair_name(e1, e2, elements):
        """Get canonical pair name (sorted by element order)."""
        i1 = elements.index(e1)
        i2 = elements.index(e2)
        if i1 <= i2:
            return f"{e1}-{e2}"
        return f"{e2}-{e1}"
