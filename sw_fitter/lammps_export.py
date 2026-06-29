import math
import os

from .models import A_from_rmin_float, B_from_rmin, GAMMA, P
from .utils import BOHR_TO_ANGSTROM, HARTREE_TO_EV, canonical_pair, canonical_triplet

Q = 0.0


def export_lammps(results_dir, dataset, params):
    eps = params["eps"]
    sw_elements = sorted({element for bond in eps for element in bond.split("-")})
    triplet_types = set(dataset.triplet_type_names)

    filepath = os.path.join(results_dir, "gnn_sw.sw")
    with open(filepath, "w") as f:
        f.write("# Stillinger-Weber v27: eps, r_min, lambda, cos_theta0 all trained\n")
        f.write(f"# SW elements: {sw_elements}\n")
        f.write("# LAMMPS pair_coeff * * gnn_sw.sw " + " ".join(sw_elements) + "\n")
        f.write("# i j k  eps sigma a lambda gamma cos0 A B p q tol\n")
        for element_i in sw_elements:
            for element_j in sw_elements:
                for element_k in sw_elements:
                    f.write(
                        _sw_line(element_i, element_j, element_k, params, triplet_types) + "\n"
                    )

    print(f"  wrote {filepath} ({len(sw_elements) ** 3} lines)")
    print(f"  LAMMPS: pair_coeff * * gnn_sw.sw {' '.join(sw_elements)}")
    return sw_elements


def _sw_line(element_i, element_j, element_k, params, triplet_types):
    eps = params["eps"]
    raw_rmin = params["raw_rmin"]
    raw_lam = params["raw_lam"]
    raw_theta0 = params["raw_theta0"]
    sigma = params["sigma"]
    cutoff = params["cutoff"]

    bond_ik = canonical_pair(element_i, element_k)
    bond_ij = canonical_pair(element_i, element_j)

    if bond_ik in eps:
        eps_val = eps[bond_ik].item()
        eps_ev = eps_val * HARTREE_TO_EV
        sigma_val = sigma[bond_ik]
        cutoff_val = cutoff[bond_ik]
        sigma_ang = sigma_val * BOHR_TO_ANGSTROM
        a = cutoff_val / sigma_val
        raw = raw_rmin[bond_ik].item()
        r_min = sigma_val + (cutoff_val - sigma_val) / (1.0 + math.exp(-raw))
        B_val = float(B_from_rmin(r_min, sigma_val, cutoff_val))
        A_val = A_from_rmin_float(r_min, sigma_val, cutoff_val)
    else:
        # Null interaction: A=0 makes V2=0 everywhere; sigma/a placeholders pass LAMMPS validation.
        eps_ev, A_val, B_val = 0.0, 0.0, 1.0
        sigma_ang, a = 2.0, 1.5

    triplet_name = canonical_triplet(element_i, element_j, element_k)
    parameterized = triplet_name in triplet_types and bond_ik in eps and bond_ij in eps
    if parameterized and eps[bond_ik].item() > 1e-12:
        lam_ij = math.exp(raw_lam[bond_ij].item())
        lam_ik = math.exp(raw_lam[bond_ik].item())
        eps_ik = eps[bond_ik].item()
        # strength in Hartree = sqrt(lam_ij * lam_ik); LAMMPS lambda = strength / eps_ik
        lam_lammps = math.sqrt(lam_ij * lam_ik) / max(abs(eps_ik), 1e-12)
        cos0 = math.tanh(raw_theta0[triplet_name].item())
    else:
        lam_lammps, cos0 = 0.0, -1.0 / 3.0

    return (
        f"{element_i} {element_j} {element_k} "
        f"{eps_ev:.10f} {sigma_ang:.10f} {a:.10f} "
        f"{lam_lammps:.10f} {GAMMA:.10f} {cos0:.10f} "
        f"{A_val:.10f} {B_val:.10f} {P:.10f} {Q:.10f} 0.0"
    )
