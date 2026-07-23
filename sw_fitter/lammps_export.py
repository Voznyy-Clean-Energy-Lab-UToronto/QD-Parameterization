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
        f.write("# Stillinger-Weber: eps, A, B, p, q, gamma, lambda, theta0 trained; cutoff (a*sigma) fixed from RDF\n")
        f.write(f"# SW elements: {sw_elements}\n")
        f.write(f"# LAMMPS pair_coeff * * {sw_filename} " + " ".join(sw_elements) + "\n")
        f.write("# i j k  eps sigma a lambda gamma cos0 A B p q tol\n")
        for element_i in sw_elements:
            for element_j in sw_elements:
                for element_k in sw_elements:
                    f.write(
                        _sw_line(
                            element_i, element_j, element_k, params, triplet_types
                        ) + "\n"
                    )

    print(f"  wrote {filepath} ({len(sw_elements)**3} lines)")
    print(f"  LAMMPS: pair_coeff * * {sw_filename} {' '.join(sw_elements)}")
    return sw_elements


def _sw_line(element_i, element_j, element_k, params, triplet_types):
    bond_ik = canonical_pair(element_i, element_k)
    bond_ij = canonical_pair(element_i, element_j)

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

    triplet_name  = canonical_triplet(element_i, element_j, element_k)
    parameterized = (triplet_name in triplet_types
                     and bond_ik in params["eps"]
                     and bond_ij in params["eps"]
                     and eps_ev > 1e-12)
    if parameterized:
        lam_triplet = math.exp(params["raw_lam"][triplet_name].item())
        eps_ij      = params["eps"][bond_ij].item()
        eps_ik      = params["eps"][bond_ik].item()
        lam_lammps  = lam_triplet * math.sqrt(eps_ij / eps_ik)
        cos0        = math.tanh(params["raw_theta0"][triplet_name].item())
    else:
        lam_lammps, cos0 = 0.0, -1.0 / 3.0

    return (
        f"{element_i} {element_j} {element_k} "
        f"{eps_ev:.10f} {sigma_ang:.10f} {a_val:.10f} "
        f"{lam_lammps:.10f} {gamma_val:.10f} {cos0:.10f} "
        f"{A_val:.10f} {B_val:.10f} {p_val:.10f} {q_val:.10f} 0.0"
    )
