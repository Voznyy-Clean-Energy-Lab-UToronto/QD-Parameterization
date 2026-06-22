"""Stillinger-Weber model — fully data-derived parameterization.

Stillinger & Weber (PRB 31, 5262, 1985) functional form, with the II-VI dimensionless
shape from Zhou et al. (PRB 88, 085309, 2013). The shape ratios (A, B, a, p, q, gamma)
are the cited SW form and are held fixed; only the physical SCALES come from the data:
  sigma  <- bond length / 1.28094   (RDF first peak r0 sets the length scale)
  cutoff <- 1.953387 * sigma        (a = cutoff/sigma, the Zhou II-VI ratio)
  A, B   <- Zhou II-VI SW shape (fixed: 7.0496, 1.116149)
  theta0 <- mean bond angle (ADF peak)
  L = lambda*eps <- angle-distribution variance (equipartition)
  eps    <- force-matching (the only force-fit parameter)

Two modes:
  naive=False (default): sigma, theta0, L, A, B are FROZEN (requires_grad off);
                         only eps is trained. This is the production fitter.
  naive=True:  sigma, theta0, L, A, B are TRAINABLE (initialised from the scan).
                         This is the ablation that shows force-matching alone
                         wrecks the physical scan values.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from .utils import HARTREE_TO_EV, BOHR_TO_ANGSTROM, inverse_softplus, canonical_pair

SW_P, SW_Q, SW_GAMMA = 4.0, 0.0, 1.20   # SW functional form (not material fits)


def build_pair_infrastructure(element_names):
    n = len(element_names)
    lookup = np.zeros((n, n), dtype=np.int64)
    names, idx = [], 0
    for i in range(n):
        for j in range(i, n):
            lookup[i, j] = lookup[j, i] = idx
            names.append(canonical_pair(element_names[i], element_names[j]))
            idx += 1
    return names, lookup


def _arctanh(x):
    x = np.clip(x, -0.999, 0.999)
    return torch.tensor(np.arctanh(x), dtype=torch.float64)


class SWModel(nn.Module):
    def __init__(self, pair_names, pair_lookup, cutoffs_bohr, triplet_type_names,
                 init_eps=None, init_sigma=None, init_cos_theta0=None, init_L=None,
                 init_A=None, init_B=None, naive=False):
        super().__init__()
        self.pair_names = pair_names
        self.triplet_type_names = triplet_type_names
        self.naive = naive
        nP, nT = len(pair_names), len(triplet_type_names)
        self.register_buffer('pair_lookup', torch.tensor(pair_lookup, dtype=torch.long))

        def _tensor(values, n, default):
            """Per-pair (or per-triplet) tensor; fall back to `default` if not supplied."""
            return torch.tensor(values if values is not None else np.full(n, default),
                                dtype=torch.float64)

        # softplus-positive parameters (stored as raw); cos_theta0 via tanh
        self.raw_eps = nn.Parameter(inverse_softplus(_tensor(init_eps, nP, 0.05)))     # always trained
        self.raw_sigma = nn.Parameter(inverse_softplus(_tensor(init_sigma, nP, 4.0)))
        self.raw_A = nn.Parameter(inverse_softplus(_tensor(init_A, nP, 7.05)))
        self.raw_B = nn.Parameter(inverse_softplus(_tensor(init_B, nP, 0.602)))
        self.raw_L = nn.Parameter(inverse_softplus(_tensor(init_L, nP, 0.05)))
        self.raw_cos_theta0 = nn.Parameter(_arctanh(
            init_cos_theta0 if init_cos_theta0 is not None else np.full(nT, -1.0 / 3.0)))

        # geometry/angular scales are FROZEN unless naive (force-matching cannot
        # constrain them; the scan does). eps is always trained.
        if not naive:
            for raw in (self.raw_sigma, self.raw_A, self.raw_B, self.raw_L, self.raw_cos_theta0):
                raw.requires_grad_(False)

        # SW form constants (always frozen) and the cutoff (geometric, frozen)
        self.register_buffer('p', _tensor(None, nP, SW_P))
        self.register_buffer('q', _tensor(None, nP, SW_Q))
        self.register_buffer('gam', _tensor(None, nP, SW_GAMMA))
        r_cut = torch.zeros(nP, dtype=torch.float64)
        for pn, cut in cutoffs_bohr.items():
            if pn in pair_names:
                r_cut[pair_names.index(pn)] = cut
        self.register_buffer('r_cut', r_cut)

    # current parameter values (with the positivity / range transforms applied)
    def _vals(self):
        return (F.softplus(self.raw_eps), F.softplus(self.raw_sigma), F.softplus(self.raw_A),
                F.softplus(self.raw_B), F.softplus(self.raw_L), torch.tanh(self.raw_cos_theta0))

    def forward(self, batch):
        n_atoms = batch.pos.size(0)
        src, tgt = batch.edge_index
        eps, sigma, A, B, L, cos_theta0 = self._vals()
        p, q, gam = self.p, self.q, self.gam

        pidx, dist, unit_vec = batch.pair_indices, batch.distances, batch.edge_unit_vectors

        # 2-body
        eps_e, sigma_e, r_cut_e = eps[pidx], sigma[pidx], self.r_cut[pidx]
        A_e, B_e, p_e, q_e = A[pidx], B[pidx], p[pidx], q[pidx]
        sig_over_r = sigma_e / dist.clamp(min=1e-10)
        in_cut = (dist < r_cut_e).to(torch.float64)
        rainv = 1.0 / (dist - r_cut_e).clamp(max=-1e-8)
        exp_cut = torch.exp(sigma_e * rainv)
        sig_r_p, sig_r_q = sig_over_r ** p_e, sig_over_r ** q_e
        poly = B_e * sig_r_p - sig_r_q
        poly_d = (p_e * B_e * sig_r_p - q_e * sig_r_q) / dist
        sf = A_e * eps_e * exp_cut * (poly_d + poly * sigma_e * rainv ** 2) * in_cut
        atom_forces = scatter_add(-sf.unsqueeze(1) * unit_vec, src, dim=0, dim_size=n_atoms)

        # 3-body. The intermediate names (facrad, facang, the coeff_* terms) deliberately
        # mirror LAMMPS pair_sw.cpp, so this can be checked line-for-line against the LAMMPS
        # source (and against consistency_check.py, which compares the two numerically).
        if hasattr(batch, 'triplet_a') and batch.triplet_a is not None and batch.triplet_a.numel() > 0:
            ta, tc, tb = batch.triplet_a, batch.triplet_center, batch.triplet_b
            pi_ca, pi_cb, tidx = batch.triplet_pi_ca, batch.triplet_pi_cb, batch.triplet_type_idx
            vec_ca, vec_cb = batch.tri_vec_ca, batch.tri_vec_cb
            cos_theta = batch.tri_cos_theta
            in_ca, in_cb = batch.tri_in_ca, batch.tri_in_cb
            rainv_ca, rainv_cb = batch.tri_rainv_ca, batch.tri_rainv_cb
            cos_rsq_ca, cos_rsq_cb = batch.tri_cos_over_rsq_ca, batch.tri_cos_over_rsq_cb
            cross_rinv = batch.tri_cross_rinv
            rrsq_ca, rrsq_cb = batch.tri_rainvsq_over_r_ca, batch.tri_rainvsq_over_r_cb

            delcs = cos_theta - cos_theta0[tidx]
            gam_sig_ca = gam[pi_ca] * sigma[pi_ca]
            gam_sig_cb = gam[pi_cb] * sigma[pi_cb]
            facexp = (torch.exp(gam_sig_ca * rainv_ca) * in_ca) * (torch.exp(gam_sig_cb * rainv_cb) * in_cb)
            lam_eps_facexp = torch.sqrt(L[pi_ca] * L[pi_cb]) * facexp   # L = lambda*eps per pair
            facrad = lam_eps_facexp * delcs ** 2
            facang = 2.0 * lam_eps_facexp * delcs
            coeff_a_ca = facrad * gam_sig_ca * rrsq_ca + facang * cos_rsq_ca
            coeff_a_cb = -facang * cross_rinv
            coeff_b_cb = facrad * gam_sig_cb * rrsq_cb + facang * cos_rsq_cb
            force_a = coeff_a_ca.unsqueeze(1) * vec_ca + coeff_a_cb.unsqueeze(1) * vec_cb
            force_b = coeff_a_cb.unsqueeze(1) * vec_ca + coeff_b_cb.unsqueeze(1) * vec_cb
            all_forces = torch.cat([force_a, force_b, -(force_a + force_b)], dim=0)
            all_idx = torch.cat([ta, tb, tc]).unsqueeze(1).expand(-1, 3)
            atom_forces = atom_forces.scatter_add(0, all_idx, all_forces)
        return atom_forces

    def params_dict(self):
        with torch.no_grad():
            eps, sigma, A, B, L, ct0 = (v.cpu().numpy() for v in self._vals())
            p, q, gam, r_cut = (b.cpu().numpy() for b in (self.p, self.q, self.gam, self.r_cut))
            pair_params = {name: {
                'eps': float(eps[i] * HARTREE_TO_EV), 'sigma': float(sigma[i] * BOHR_TO_ANGSTROM),
                'A': float(A[i]), 'B': float(B[i]), 'p': float(p[i]), 'q': float(q[i]),
                'gamma': float(gam[i]),
                'L': float(L[i] * HARTREE_TO_EV),
                'lambda': float(L[i] / eps[i]) if eps[i] > 1e-12 else 0.0,
                'cutoff': float(r_cut[i] * BOHR_TO_ANGSTROM),
                'a': float(r_cut[i] / sigma[i]) if sigma[i] > 1e-9 else 0.0,
            } for i, name in enumerate(self.pair_names)}
            triplet_params = {name: {'cos_theta0': float(ct0[i])}
                              for i, name in enumerate(self.triplet_type_names)}
            return pair_params, triplet_params
