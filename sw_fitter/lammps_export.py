import os

from .models import GAMMA, P
from .utils import BOHR_TO_ANGSTROM, HARTREE_TO_EV, canonical_pair, canonical_triplet

Q = 0.0  # the SW q exponent is 0 for this form


def export_lammps(results_dir, dataset, params):
    eps = params["eps"]
    sw_elements = sorted({element for bond in eps for element in bond.split("-")})
    triplet_types = set(dataset.triplet_type_names)

    filepath = os.path.join(results_dir, "gnn_sw.sw")
    with open(filepath, "w") as f:
        f.write(f"# Stillinger-Weber, data-derived. SW elements: {sw_elements}\n")
        f.write("# i j k  eps sigma a lambda gamma cos0 A B p q tol\n")
        for element_i in sw_elements:
            for element_j in sw_elements:
                for element_k in sw_elements:
                    f.write(
                        _sw_line(element_i, element_j, element_k, params, triplet_types)
                        + "\n"
                    )

    print(
        f"  wrote {filepath} ({len(sw_elements) ** 3} lines, SW elements {sw_elements})"
    )
    return sw_elements


def _sw_line(element_i, element_j, element_k, params, triplet_types):
    eps, sigma, cutoff = params["eps"], params["sigma"], params["cutoff"]
    A, B, L, cos_theta0 = params["A"], params["B"], params["L"], params["cos_theta0"]

    bond_ik = canonical_pair(element_i, element_k)
    bond_ij = canonical_pair(element_i, element_j)

    if bond_ik in eps:
        eps_ev = eps[bond_ik].item() * HARTREE_TO_EV
        sigma_ang = sigma[bond_ik] * BOHR_TO_ANGSTROM
        a = cutoff[bond_ik] / sigma[bond_ik]
        A_value, B_value = A[bond_ik], B[bond_ik]
    else:
        eps_ev = sigma_ang = a = A_value = B_value = 0.0

    triplet_name = canonical_triplet(element_i, element_j, element_k)
    parameterized = triplet_name in triplet_types and bond_ik in eps and bond_ij in eps
    if parameterized and eps[bond_ik].item() > 1e-12:
        lam = (L[bond_ij] * L[bond_ik]) ** 0.5 / eps[bond_ik].item()
        cos0 = cos_theta0[triplet_name]
    else:
        lam, cos0 = 0.0, -1.0 / 3.0

    return (
        f"{element_i} {element_j} {element_k} "
        f"{eps_ev:.10f} {sigma_ang:.10f} {a:.10f} "
        f"{lam:.10f} {GAMMA:.10f} {cos0:.10f} "
        f"{A_value:.10f} {B_value:.10f} {P:.10f} {Q:.10f} 0.0"
    )
