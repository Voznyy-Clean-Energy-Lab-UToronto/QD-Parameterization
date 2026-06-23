import math

import torch

from .utils import canonical_pair

P = 4.0  # the (sigma/r)^p exponent
GAMMA = 1.20  # the 3-body exponential decay
# (q = 0, so the second 2-body term (sigma/r)^q is just the constant 1


def sw_2body_force(bond_length, eps, sigma, A, B, cutoff):
    sigma_over_r = sigma / bond_length
    bracket = B * sigma_over_r**P - 1.0  # the ( B(sigma/r)^4 - 1 ) factor
    d_bracket = P * B * sigma_over_r**P / bond_length  # = -d(bracket)/dr
    inv_gap = 1.0 / (bond_length - cutoff)  # 1/(r - cutoff); negative inside cutoff
    decay = torch.exp(sigma * inv_gap)  # the exp( sigma/(r-cutoff) ) cutoff factor
    return A * eps * decay * (d_bracket + bracket * sigma * inv_gap**2)


def sw_3body_forces(
    vec_ca,
    vec_cb,
    strength,
    cos_theta0,
    gamma_sigma_ca,
    gamma_sigma_cb,
    cutoff_ca,
    cutoff_cb,
):
    length_ca = vec_ca.norm(dim=1)
    length_cb = vec_cb.norm(dim=1)
    cos_theta = (vec_ca * vec_cb).sum(dim=1) / (length_ca * length_cb)

    gap_ca = (length_ca - cutoff_ca).clamp(max=-1e-9)  # negative inside the cutoff
    gap_cb = (length_cb - cutoff_cb).clamp(max=-1e-9)
    decay_ca = torch.exp(gamma_sigma_ca / gap_ca)
    decay_cb = torch.exp(gamma_sigma_cb / gap_cb)
    energy_scale = strength * decay_ca * decay_cb  # everything except the angle term

    angle_term = cos_theta - cos_theta0
    radial = energy_scale * angle_term**2  # the part that pushes along the bonds
    angular = 2.0 * energy_scale * angle_term  # the part that opens/closes the angle

    decay_slope_ca = gamma_sigma_ca / gap_ca**2 / length_ca
    decay_slope_cb = gamma_sigma_cb / gap_cb**2 / length_cb
    cos_over_lensq_ca = cos_theta / length_ca**2
    cos_over_lensq_cb = cos_theta / length_cb**2
    cross = 1.0 / (length_ca * length_cb)

    along_ca = radial * decay_slope_ca + angular * cos_over_lensq_ca
    along_cb = radial * decay_slope_cb + angular * cos_over_lensq_cb
    mixed = -angular * cross

    force_a = along_ca.unsqueeze(1) * vec_ca + mixed.unsqueeze(1) * vec_cb
    force_b = mixed.unsqueeze(1) * vec_ca + along_cb.unsqueeze(1) * vec_cb
    return force_a, force_b


def two_body_forces(positions, edges_by_type, params):
    eps, sigma = params["eps"], params["sigma"]
    A, B, cutoff = params["A"], params["B"], params["cutoff"]
    forces = torch.zeros_like(positions)
    for bond in edges_by_type:
        atom_i, atom_j = edges_by_type[bond]
        bond_vector = positions[atom_j] - positions[atom_i]
        bond_length = bond_vector.norm(dim=1)
        force_magnitude = sw_2body_force(
            bond_length, eps[bond], sigma[bond], A[bond], B[bond], cutoff[bond]
        )
        bond_force = (force_magnitude / bond_length).unsqueeze(1) * bond_vector
        # index_add(dim, indices, values) adds 'values' onto the rows named by 'indices'.
        forces = forces.index_add(
            0, atom_i, -bond_force
        )  # Newton's third law: atom i and
        forces = forces.index_add(
            0, atom_j, bond_force
        )  # atom j feel equal, opposite forces
    return forces


def three_body_forces(positions, triplets_by_type, params):
    sigma, cutoff, strength_of = params["sigma"], params["cutoff"], params["L"]
    cos_theta0_of = params["cos_theta0"]
    forces = torch.zeros_like(positions)
    for triplet_name in triplets_by_type:
        centre_idx, a_idx, b_idx = triplets_by_type[triplet_name]
        # the triplet name encodes the centre and its two leg bonds, e.g.
        # 'Cd_core:Se_core-Se_surf' -> centre Cd_core, legs Cd_core-Se_core, Cd_core-Se_surf
        centre, legs = triplet_name.split(":")
        leg_a_subtype, leg_b_subtype = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a_subtype)
        bond_cb = canonical_pair(centre, leg_b_subtype)
        strength = math.sqrt(
            strength_of[bond_ca] * strength_of[bond_cb]
        )  # Zhou lambda*eps mixing

        force_a, force_b = sw_3body_forces(
            positions[a_idx] - positions[centre_idx],
            positions[b_idx] - positions[centre_idx],
            strength,
            cos_theta0_of[triplet_name],
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
