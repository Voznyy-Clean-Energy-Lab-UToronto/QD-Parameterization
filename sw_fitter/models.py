import torch
import torch.nn.functional as F

from .utils import canonical_pair

SIGMA_RATIO_MIN, SIGMA_RATIO_MAX = 1.0, 2.0
SIGMA_FIXED_POINT_ITERS = 4


def sw_sigma_ratio(B, a, p, q, n=100):
    B = float(B); a = float(a); p = float(p); q = float(q)
    x = torch.linspace(1.001, a - 1e-3, n, dtype=torch.float64)
    ratio = float(x[torch.argmin((B * x ** (-p) - x ** (-q)) * torch.exp(1.0 / (x - a)))])
    return min(max(ratio, SIGMA_RATIO_MIN), SIGMA_RATIO_MAX)


def bond_sigma(bond, params):
    B = F.softplus(params["raw_B"][bond])
    p = params["raw_p"][bond]
    q = params["raw_q"][bond]
    cutoff_over_r0 = float(params["cutoff"][bond] / params["r0"][bond])
    sigma_ratio = 1.12
    for _ in range(SIGMA_FIXED_POINT_ITERS):
        sigma_ratio = sw_sigma_ratio(B, cutoff_over_r0 * sigma_ratio, p, q)
    return params["r0"][bond] / sigma_ratio


def sw_2body_force(bond_length, eps, A, B, p, q, sigma, cutoff):
    sig_over_r = sigma / bond_length
    bracket    = B * sig_over_r ** p - sig_over_r ** q
    d_bracket  = (p * B * sig_over_r ** p - q * sig_over_r ** q) / bond_length
    inv_gap    = 1.0 / (bond_length - cutoff).clamp(max=-1e-9)
    decay      = torch.exp(sigma * inv_gap)
    return A * eps * decay * (d_bracket + bracket * sigma * inv_gap ** 2)


def sw_3body_forces(vec_ca, vec_cb, length_ca, length_cb, cos_theta,
                    strength, cos_theta0, gamma_sigma_ca, gamma_sigma_cb, cutoff_ca, cutoff_cb):
    gap_ca = (length_ca - cutoff_ca).clamp(max=-1e-9)
    gap_cb = (length_cb - cutoff_cb).clamp(max=-1e-9)
    energy_scale = strength * torch.exp(gamma_sigma_ca / gap_ca) * torch.exp(gamma_sigma_cb / gap_cb)

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


def two_body_forces(graph, params, sigma_by_bond):
    positions = graph["positions"]
    forces = torch.zeros_like(positions)
    for bond, (atom_i, atom_j) in graph["edges"].items():
        sigma = sigma_by_bond[bond]
        bond_length = graph["edge_len"][bond]
        force_magnitude = sw_2body_force(
            bond_length,
            params["eps"][bond],
            torch.exp(params["raw_A"][bond]),
            F.softplus(params["raw_B"][bond]),
            params["raw_p"][bond],
            params["raw_q"][bond],
            sigma,
            params["cutoff"][bond],
        )
        bond_vector = positions[atom_j] - positions[atom_i]
        bond_force = (force_magnitude / bond_length).unsqueeze(1) * bond_vector
        forces = forces.index_add(0, atom_i, -bond_force).index_add(0, atom_j, bond_force)
    return forces


def three_body_forces(graph, params, sigma_by_bond):
    positions = graph["positions"]
    forces = torch.zeros_like(positions)
    for triplet_name, (centre_idx, a_idx, b_idx) in graph["triplets"].items():
        centre, legs = triplet_name.split(":")
        leg_a, leg_b = legs.split("-")
        bond_ca = canonical_pair(centre, leg_a)
        bond_cb = canonical_pair(centre, leg_b)

        strength = torch.exp(params["raw_lam"][triplet_name]) * torch.sqrt(
            params["eps"][bond_ca] * params["eps"][bond_cb]
        )
        cos_theta0 = torch.tanh(params["raw_theta0"][triplet_name])
        gamma_ca = F.softplus(params["raw_gamma"][bond_ca])
        gamma_cb = F.softplus(params["raw_gamma"][bond_cb])
        sigma_ca = sigma_by_bond[bond_ca]
        sigma_cb = sigma_by_bond[bond_cb]

        length_ca, length_cb, cos_theta = graph["tri_len"][triplet_name]
        vec_ca = positions[a_idx] - positions[centre_idx]
        vec_cb = positions[b_idx] - positions[centre_idx]

        force_a, force_b = sw_3body_forces(
            vec_ca, vec_cb, length_ca, length_cb, cos_theta,
            strength, cos_theta0,
            gamma_ca * sigma_ca, gamma_cb * sigma_cb,
            params["cutoff"][bond_ca], params["cutoff"][bond_cb],
        )
        forces = forces.index_add(0, a_idx, force_a).index_add(0, b_idx, force_b) \
                       .index_add(0, centre_idx, -(force_a + force_b))
    return forces


def sw_forces(graph, params):
    sigma_by_bond = {bond: bond_sigma(bond, params) for bond in params["eps"]}
    return two_body_forces(graph, params, sigma_by_bond) + three_body_forces(graph, params, sigma_by_bond)
