import numpy as np
import pyscal3 as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch_scatter import scatter_add

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    inverse_softplus, canonical_pair, base_element,
)


def compute_q6(positions, elements, inorganic_elements, cutoff=5.5):
    inorganic = set(inorganic_elements)
    inorg_mask = np.array([e in inorganic for e in elements])
    inorg_pos = positions[inorg_mask]
    inorg_species = [elements[i] for i in range(len(elements)) if inorg_mask[i]]

    sys = pc.System(
        atoms={'positions': inorg_pos, 'species': inorg_species},
        box=[[100, 0, 0], [0, 100, 0], [0, 0, 100]]
    )
    sys.find.neighbors(method='cutoff', cutoff=cutoff)
    sys.calculate.steinhardt_parameter(6)

    q6 = np.zeros(len(elements))
    q6[inorg_mask] = np.array(sys.atoms['q6'])
    return q6


def assign_types(symbols, positions_ang, inorganic_elements, organic_cutoff=3.0):
    elements = list(symbols)
    inorganic = set(inorganic_elements)
    n = len(elements)

    q6 = compute_q6(positions_ang, elements, inorganic_elements)

    dmat = cdist(positions_ang, positions_ang)
    np.fill_diagonal(dmat, np.inf)
    org_cn = np.zeros(n, dtype=int)
    for i in range(n):
        if elements[i] not in inorganic:
            continue
        for j in range(n):
            if elements[j] == 'O' and dmat[i, j] < organic_cutoff:
                org_cn[i] += 1

    # classify: slig first, then q6 gap on the rest
    result = list(symbols)
    for elem in sorted(inorganic):
        idx = np.where(np.array(elements) == elem)[0]
        if len(idx) == 0:
            continue

        # step 1: ligand-bound atoms
        slig_mask = org_cn[idx] >= 1
        for ai in idx[slig_mask]:
            result[ai] = f"{elem}_slig"

        # step 2: q6 gap detection on non-slig atoms
        non_slig = idx[~slig_mask]
        if len(non_slig) < 2:
            for ai in non_slig:
                result[ai] = f"{elem}_core"
            continue

        q6_vals = np.sort(q6[non_slig])
        gaps = np.diff(q6_vals)
        biggest_gap = np.argmax(gaps)
        threshold = 0.5 * (q6_vals[biggest_gap] + q6_vals[biggest_gap + 1])

        for ai in non_slig:
            if q6[ai] <= threshold:
                result[ai] = f"{elem}_core"
            else:
                result[ai] = f"{elem}_surf"

    return result, q6, org_cn


def build_type_pair_infrastructure(type_names):
    n = len(type_names)
    cpair_lookup = np.zeros((n, n), dtype=np.int64)
    cpair_names = []
    cpair_to_base = {}
    idx = 0
    for i in range(n):
        for j in range(i, n):
            cpair_lookup[i, j] = idx
            cpair_lookup[j, i] = idx
            cp = canonical_pair(type_names[i], type_names[j])
            cpair_names.append(cp)
            cpair_to_base[cp] = canonical_pair(base_element(type_names[i]),
                                               base_element(type_names[j]))
            idx += 1
    return cpair_names, cpair_lookup, cpair_to_base


class MorseModel(nn.Module):
    def __init__(self, cpair_names, cpair_lookup, init_D_e, init_alpha, init_r0):
        super().__init__()
        self.cpair_names = cpair_names
        self.register_buffer('cpair_lookup', torch.tensor(cpair_lookup, dtype=torch.long))

        self.raw_D_e = nn.Parameter(inverse_softplus(torch.tensor(init_D_e, dtype=torch.float64)))
        self.raw_alpha = nn.Parameter(inverse_softplus(torch.tensor(init_alpha, dtype=torch.float64)))
        self.raw_r0 = nn.Parameter(torch.tensor(init_r0, dtype=torch.float64))

    def forward(self, batch):
        D_e = F.softplus(self.raw_D_e)[batch.pair_indices]
        alpha = F.softplus(self.raw_alpha)[batch.pair_indices]
        r0 = self.raw_r0[batch.pair_indices]

        x = batch.distances - r0
        exp_term = torch.exp(-alpha * x)
        scalar_force = 2.0 * D_e * alpha * (exp_term ** 2 - exp_term)

        force_vecs = -scalar_force.unsqueeze(1) * batch.edge_unit_vectors
        return scatter_add(force_vecs, batch.edge_index[0], dim=0, dim_size=batch.pos.size(0))

    def get_type_pair_params(self):
        with torch.no_grad():
            D_e = F.softplus(self.raw_D_e).cpu().numpy()
            alpha = F.softplus(self.raw_alpha).cpu().numpy()
            r0 = self.raw_r0.cpu().numpy()
            return {name: {'D_e': float(D_e[i] * HARTREE_TO_EV),
                           'alpha': float(alpha[i] / BOHR_TO_ANGSTROM),
                           'r0': float(r0[i] * BOHR_TO_ANGSTROM)}
                    for i, name in enumerate(self.cpair_names)}
