import math

import torch

from .utils import canonical_pair

P = 4.0       # (sigma/r)^p exponent — fixed from SW literature
GAMMA = 1.20  # 3-body exponential decay coefficient
Q = 0.0       # second 2-body exponent

SIGMA_RATIO = 1.28094    # r0 / sigma  (location of V2 minimum in standard SW)
CUTOFF_RATIO = 1.953387  # cutoff / sigma  (the LAMMPS 'a' parameter)


def rmin_from_raw(raw_rmin, sigma, cutoff):
    return sigma + (cutoff - sigma) * torch.sigmoid(raw_rmin)


def B_from_rmin(r_min, sigma, cutoff):
    ratio4 = (r_min / sigma) ** P
    denom = 1.0 + 4.0 * (r_min - cutoff) ** 2 / (sigma * r_min)
    return ratio4 / denom


def A_from_rmin(r_min, sigma, cutoff):
    B = B_from_rmin(r_min, sigma, cutoff)
    bracket_at_rmin = B * (sigma / r_min) ** P - 1.0  # always < 0
    decay_at_rmin = torch.exp(sigma / (r_min - cutoff))  # r_min < cutoff so exponent < 0
    return -1.0 / (bracket_at_rmin * decay_at_rmin)  # always > 0


def A_from_rmin_float(r_min, sigma, cutoff):
    B = B_from_rmin(r_min, sigma, cutoff)
    bracket_at_rmin = B * (sigma / r_min) ** P - 1.0
    decay_at_rmin = math.exp(sigma / (r_min - cutoff))
    return -1.0 / (bracket_at_rmin * decay_at_rmin)


def sw_2body_force(bond_length, eps, raw_rmin, sigma, cutoff):
    r_min = rmin_from_raw(raw_rmin, sigma, cutoff)
    B = B_from_rmin(r_min, sigma, cutoff)
    A = A_from_rmin(r_min, sigma, cutoff)
    sigma_over_r = sigma / bond_length
    bracket = B * sigma_over_r ** P - 1.0
    d_bracket = P * B * sigma_over_r ** P / bond_length
    inv_gap = 1.0 / (bond_length - cutoff)
    decay = torch.exp(sigma * inv_gap)
    return A * eps * decay * (d_bracket + bracket * sigma * inv_gap ** 2)


def sw_3body_forces(
    vec_ca, vec_cb,
    strength, cos_theta0,
    gamma_sigma_ca, gamma_sigma_cb,
    cutoff_ca, cutoff_cb,
):
    length_ca = vec_ca.norm(dim=1)
    length_cb = vec_cb.norm(dim=1)
    cos_theta = (vec_ca * vec_cb).sum(dim=1) / (length_ca * length_cb)

    gap_ca = (length_ca - cutoff_ca).clamp(max=-1e-9)
    gap_cb = (length_cb - cutoff_cb).clamp(max=-1e-9)
    decay_ca = torch.exp(gamma_sigma_ca / gap_ca)
    decay_cb = torch.exp(gamma_sigma_cb / gap_cb)
    energy_scale = strength * decay_ca * decay_cb

    angle_term = cos_theta - cos_theta0
    radial = energy_scale * angle_term ** 2
    angular = 2.0 * energy_scale * angle_term

    decay_slope_ca = gamma_sigma_ca / gap_ca ** 2 / length_ca
    decay_slope_cb = gamma_sigma_cb / gap_cb ** 2 / length_cb
    cos_over_lensq_ca = cos_theta / length_ca ** 2
    cos_over_lensq_cb = cos_theta / length_cb ** 2
    cross = 1.0 / (length_ca * length_cb)

    along_ca = radial * decay_slope_ca + angular * cos_over_lensq_ca
    along_cb = radial * decay_slope_cb + angular * cos_over_lensq_cb
    mixed = -angular * cross

    force_a = along_ca.unsqueeze(1) * vec_ca + mixed.unsqueeze(1) * vec_cb
    force_b = mixed.unsqueeze(1) * vec_ca + along_cb.unsqueeze(1) * vec_cb
    return force_a, force_b


def two_body_forces(positions, edges_by_type, params):
    eps = params["eps"]
    raw_rmin = params["raw_rmin"]
    sigma, cutoff = params["sigma"], params["cutoff"]
    forces = torch.zeros_like(positions)
    for bond in edges_by_type:
        atom_i, atom_j = edges_by_type[bond]
        bond_vector = positions[atom_j] - positions[atom_i]
        bond_length = bond_vector.norm(dim=1)
        force_magnitude = sw_2body_force(
            bond_length, eps[bond], raw_rmin[bond], sigma[bond], cutoff[bond]
        )
        bond_force = (force_magnitude / bond_length).unsqueeze(1) * bond_vector
        forces = forces.index_add(0, atom_i, -bond_force)
        forces = forces.index_add(0, atom_j, bond_force)
    return forces


def three_body_forces(positions, triplets_by_type, params):
    sigma = params["sigma"]
    cutoff = params["cutoff"]
    raw_lam = params["raw_lam"]
    raw_theta0 = params["raw_theta0"]
    forces = torch.zeros_like(positions)
    for triplet_name in triplets_by_type:
        centre_idx, a_idx, b_idx = triplets_by_type[triplet_name]
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        lam_ca = torch.exp(raw_lam[bond_ca])
        lam_cb = torch.exp(raw_lam[bond_cb])
        strength = torch.sqrt(lam_ca * lam_cb)

        cos_theta0 = torch.tanh(raw_theta0[triplet_name])

        force_a, force_b = sw_3body_forces(
            positions[a_idx] - positions[centre_idx],
            positions[b_idx] - positions[centre_idx],
            strength,
            cos_theta0,
            GAMMA * sigma[bond_ca],
            GAMMA * sigma[bond_cb],
            cutoff[bond_ca],
            cutoff[bond_cb],
        )
        forces = forces.index_add(0, a_idx, force_a)
        forces = forces.index_add(0, b_idx, force_b)
        forces = forces.index_add(0, centre_idx, -(force_a + force_b))
    return forces


def sw_forces(graph, params):
    two = two_body_forces(graph["positions"], graph["edges"], params)
    three = three_body_forces(graph["positions"], graph["triplets"], params)
    return two + three
