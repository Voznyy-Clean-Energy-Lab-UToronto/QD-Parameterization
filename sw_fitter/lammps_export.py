"""LAMMPS .sw export — Zhou et al. (PRB 88, 085309, 2013) per-pair-lambda mixing.

The three-body strength is generated from per-pair lambda by the Zhou rule
    lambda_ijk = sqrt(lambda_ij eps_ij lambda_ik eps_ik) / eps_ik,   eps_ijk = eps_ik,
and the equilibrium angle is the ideal tetrahedral cos_theta0 = -1/3 for every triplet.
The .sw file lists only the SW-active elements (those in a scoped pair, e.g. Cd/Se/O);
organic atoms (C, H) are mapped to NULL in pair_coeff and handled by CHARMM at MD time.
"""
import os
import math
from .utils import canonical_pair
from .data import canonical_triplet


def export_lammps(output_dir, pair_params, triplet_params, elements, original_data=None):
    os.makedirs(output_dir, exist_ok=True)
    sw_elems = sorted({e for pk, pp in pair_params.items()
                       if pp.get('cutoff', 0) > 0.01 and pp.get('eps', 0) > 1e-9
                       for e in pk.split('-')})
    sw_file = os.path.join(output_dir, 'gnn_sw.sw')
    _write_sw_file(sw_file, pair_params, triplet_params, sw_elems)
    print(f'  SW export: {sw_file}  (SW elements: {sw_elems})')
    return sw_elems


def _get(pair_params, a, b):
    pp = pair_params.get(canonical_pair(a, b))
    if pp and pp.get('cutoff', 0) > 0.01 and pp.get('eps', 0) > 1e-9:
        return pp
    return None


def _write_sw_file(filepath, pair_params, triplet_params, elements):
    """One line per ordered triplet (i,j,k). 2-body params for pair i-j are read by
    LAMMPS from the i-j-j line; the i-j-k line carries the 3-body lambda/eps/cos0.
    cos_theta0 is the data-derived mean bond angle (per triplet)."""
    n = len(elements)
    with open(filepath, 'w') as f:
        f.write(f"# Stillinger-Weber, data-derived (sigma,theta0,L from data). Elements: {elements}\n")
        f.write("# eps sigma a lambda gamma cos0 A B p q tol\n")
        for ei in elements:
            for ej in elements:
                for ek in elements:
                    pik = _get(pair_params, ei, ek)     # 2-body of the i-k pair
                    pij = _get(pair_params, ei, ej)     # for the 3-body mixing
                    if pik:
                        eps_ik = pik['eps']; sig = pik['sigma']
                        a = pik['cutoff'] / sig
                        Av, Bv, pv, qv, gam = pik['A'], pik['B'], pik['p'], pik['q'], pik['gamma']
                    else:
                        eps_ik = sig = a = Av = Bv = pv = qv = gam = 0.0
                    # three-body: lambda_ijk*eps_ijk = sqrt(L_ij L_ik), eps_ijk=eps_ik
                    # (L = lambda*eps per pair) -> lambda_ijk = sqrt(L_ij L_ik)/eps_ik
                    if pik and pij and eps_ik > 1e-9:
                        lam_line = math.sqrt(pij['L'] * pik['L']) / eps_ik
                    else:
                        lam_line = 0.0
                    tp = triplet_params.get(canonical_triplet(ei, ej, ek))
                    cos0 = tp['cos_theta0'] if tp else -1.0 / 3.0   # data-derived mean angle
                    f.write(f"{ei} {ej} {ek} "
                            f"{eps_ik:.10f} {sig:.10f} {a:.10f} "
                            f"{lam_line:.10f} {gam:.10f} {cos0:.10f} "
                            f"{Av:.10f} {Bv:.10f} {pv:.10f} {qv:.10f} 0.0\n")
    print(f"  Wrote {filepath} ({n**3} lines, {n} SW elements)")
