import torch
import torch.nn.functional as F

from .utils import canonical_pair


def sw_2body_force(bond_length, eps, A, B, p, q, sigma, cutoff):
    sig_over_r = sigma / bond_length
    bracket    = B * sig_over_r**p - sig_over_r**q
    d_bracket  = (p * B * sig_over_r**p - q * sig_over_r**q) / bond_length
    inv_gap    = 1.0 / (bond_length - cutoff)   # negative because r < cutoff always
    decay      = torch.exp(sigma * inv_gap)
    return A * eps * decay * (d_bracket + bracket * sigma * inv_gap**2)


def sw_3body_forces(
    vec_ca, vec_cb,
    strength, cos_theta0,
    gamma_sigma_ca, gamma_sigma_cb,
    cutoff_ca, cutoff_cb,
):
    length_ca = vec_ca.norm(dim=1)
    length_cb = vec_cb.norm(dim=1)
    cos_theta  = (vec_ca * vec_cb).sum(dim=1) / (length_ca * length_cb)

    gap_ca = (length_ca - cutoff_ca).clamp(max=-1e-9)
    gap_cb = (length_cb - cutoff_cb).clamp(max=-1e-9)
    decay_ca     = torch.exp(gamma_sigma_ca / gap_ca)
    decay_cb     = torch.exp(gamma_sigma_cb / gap_cb)
    energy_scale = strength * decay_ca * decay_cb

    angle_term = cos_theta - cos_theta0
    radial     = energy_scale * angle_term ** 2
    angular    = 2.0 * energy_scale * angle_term

    decay_slope_ca    = gamma_sigma_ca / gap_ca ** 2 / length_ca
    decay_slope_cb    = gamma_sigma_cb / gap_cb ** 2 / length_cb
    cos_over_lensq_ca = cos_theta / length_ca ** 2
    cos_over_lensq_cb = cos_theta / length_cb ** 2
    cross = 1.0 / (length_ca * length_cb)

    along_ca = radial * decay_slope_ca + angular * cos_over_lensq_ca
    along_cb = radial * decay_slope_cb + angular * cos_over_lensq_cb
    mixed    = -angular * cross

    force_a = along_ca.unsqueeze(1) * vec_ca + mixed.unsqueeze(1) * vec_cb
    force_b = mixed.unsqueeze(1) * vec_ca + along_cb.unsqueeze(1) * vec_cb
    return force_a, force_b


def two_body_forces(positions, edges_by_type, params):
    forces = torch.zeros_like(positions)
    for bond in edges_by_type:
        atom_i, atom_j = edges_by_type[bond]
        bond_vector = positions[atom_j] - positions[atom_i]
        bond_length = bond_vector.norm(dim=1)

        A = torch.exp(params["raw_A"][bond])
        B = F.softplus(params["raw_B"][bond])
        p = params["raw_p"][bond]     # unconstrained
        q = params["raw_q"][bond]     # unconstrained

        force_magnitude = sw_2body_force(
            bond_length,
            params["eps"][bond],
            A, B, p, q,
            params["sigma"][bond],
            params["cutoff"][bond],
        )
        bond_force = (force_magnitude / bond_length).unsqueeze(1) * bond_vector
        forces = forces.index_add(0, atom_i, -bond_force)
        forces = forces.index_add(0, atom_j,  bond_force)
    return forces


def three_body_forces(positions, triplets_by_type, params):
    forces = torch.zeros_like(positions)
    for triplet_name in triplets_by_type:
        centre_idx, a_idx, b_idx = triplets_by_type[triplet_name]
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        lam_ca     = torch.exp(params["raw_lam"][bond_ca])
        lam_cb     = torch.exp(params["raw_lam"][bond_cb])
        strength   = torch.sqrt(lam_ca * lam_cb)
        cos_theta0 = torch.tanh(params["raw_theta0"][triplet_name])
        gamma_ca   = F.softplus(params["raw_gamma"][bond_ca])
        gamma_cb   = F.softplus(params["raw_gamma"][bond_cb])

        force_a, force_b = sw_3body_forces(
            positions[a_idx]      - positions[centre_idx],
            positions[b_idx]      - positions[centre_idx],
            strength, cos_theta0,
            gamma_ca * params["sigma"][bond_ca],
            gamma_cb * params["sigma"][bond_cb],
            params["cutoff"][bond_ca],
            params["cutoff"][bond_cb],
        )
        forces = forces.index_add(0, a_idx,       force_a)
        forces = forces.index_add(0, b_idx,       force_b)
        forces = forces.index_add(0, centre_idx, -(force_a + force_b))
    return forces


def sw_forces(graph, params):
    two   = two_body_forces(  graph["positions"], graph["edges"],    params)
    three = three_body_forces(graph["positions"], graph["triplets"], params)
    return two + three
