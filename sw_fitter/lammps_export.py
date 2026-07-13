import math
import os

import torch.nn.functional as F

from .models import bond_sigma
from .utils import BOHR_TO_ANGSTROM, HARTREE_TO_EV, canonical_pair, canonical_triplet


def export_lammps(results_dir, dataset, params, formula):
    eps         = params["eps"]
    sw_elements = sorted({element for bond in eps for element in bond.split("-")})
    triplet_types = set(dataset.triplet_type_names)

    sw_filename = f"{formula}.sw"
    filepath    = os.path.join(results_dir, sw_filename)
    with open(filepath, "w") as f:
        f.write("# Stillinger-Weber v33: A, B, p, q, gamma, eps, lambda, theta0, a all trained\n")
        f.write(f"# SW elements: {sw_elements}\n")
        f.write(f"# LAMMPS pair_coeff * * {sw_filename} " + " ".join(sw_elements) + "\n")
        f.write("# i j k  eps sigma a lambda gamma cos0 A B p q tol\n")
        for ei in sw_elements:
            for ej in sw_elements:
                for ek in sw_elements:
                    f.write(_sw_line(ei, ej, ek, params, triplet_types) + "\n")

    print(f"  wrote {filepath} ({len(sw_elements)**3} lines)")
    print(f"  LAMMPS: pair_coeff * * {sw_filename} {' '.join(sw_elements)}")
    return sw_elements


def _sw_line(ei, ej, ek, params, triplet_types):
    bond_ik = canonical_pair(ei, ek)
    bond_ij = canonical_pair(ei, ej)

    if bond_ik in params["eps"]:
        eps_ev    = params["eps"][bond_ik].item() * HARTREE_TO_EV
        sigma_bohr = bond_sigma(bond_ik, params)
        sigma_ang = sigma_bohr * BOHR_TO_ANGSTROM
        a_val     = float(params["cutoff"][bond_ik] / sigma_bohr)
        A_val     = math.exp(params["raw_A"][bond_ik].item())
        B_val     = F.softplus(params["raw_B"][bond_ik]).item()
        p_val     = params["raw_p"][bond_ik].item()
        q_val     = params["raw_q"][bond_ik].item()
        gamma_val = F.softplus(params["raw_gamma"][bond_ik]).item()
    else:
        eps_ev, A_val, B_val = 0.0, 0.0, 1.0
        sigma_ang, a_val    = 2.0, 1.5
        p_val, q_val, gamma_val = 4.0, 0.0, 1.2

    triplet_name  = canonical_triplet(ei, ej, ek)
    parameterized = (triplet_name in triplet_types
                     and bond_ik in params["eps"]
                     and bond_ij in params["eps"]
                     and eps_ev > 1e-12)
    if parameterized:
        lam_ij     = math.exp(params["raw_lam"][bond_ij].item())
        lam_ik     = math.exp(params["raw_lam"][bond_ik].item())
        eps_ij     = params["eps"][bond_ij].item()
        eps_ik     = params["eps"][bond_ik].item()
        lam_lammps = math.sqrt(lam_ij * eps_ij * lam_ik * eps_ik) / max(abs(eps_ik), 1e-12)
        cos0       = math.tanh(params["raw_theta0"][triplet_name].item())
    else:
        lam_lammps, cos0 = 0.0, -1.0 / 3.0

    return (
        f"{ei} {ej} {ek} "
        f"{eps_ev:.10f} {sigma_ang:.10f} {a_val:.10f} "
        f"{lam_lammps:.10f} {gamma_val:.10f} {cos0:.10f} "
        f"{A_val:.10f} {B_val:.10f} {p_val:.10f} {q_val:.10f} 0.0"
    )
