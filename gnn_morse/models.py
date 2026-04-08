import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch_scatter import scatter_add

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    inverse_softplus,
)


class VectorQuantizer(nn.Module):

    def __init__(self, elements, embed_dim, max_codes_per_element=4,
                 core_elements=None, ema_decay=0.99, commitment_weight=0.25):
        super().__init__()
        self.elements = elements
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight

        # Only core elements get multiple codes; everything else gets 1
        core_set = set(core_elements) if core_elements else set(elements)
        self.n_codes_per_elem = {}
        for elem in elements:
            self.n_codes_per_elem[elem] = max_codes_per_element if elem in core_set else 1

        # Flat codebook: all element codes concatenated
        self.elem_offset = {}
        total_codes = 0
        for elem in elements:
            self.elem_offset[elem] = total_codes
            total_codes += self.n_codes_per_elem[elem]
        self.total_codes = total_codes

        self.register_buffer('codebook', torch.zeros(total_codes, embed_dim, dtype=torch.float64))
        self.register_buffer('ema_count', torch.zeros(total_codes, dtype=torch.float64))
        self.register_buffer('ema_embed_sum', torch.zeros(total_codes, embed_dim, dtype=torch.float64))
        self.register_buffer('initialized', torch.tensor(False))

        # Precompute element -> code index ranges for vectorized forward
        self._elem_ranges = []
        for elem in elements:
            start = self.elem_offset[elem]
            end = start + self.n_codes_per_elem[elem]
            self._elem_ranges.append((start, end))

    @torch.no_grad()
    def initialize_codebook(self, h, elem_indices):
        for ei, elem in enumerate(self.elements):
            mask = (elem_indices == ei)
            if mask.sum() == 0:
                continue
            start = self.elem_offset[elem]
            n_codes = self.n_codes_per_elem[elem]
            end = start + n_codes
            elem_h = h[mask]

            if n_codes == 1:
                self.codebook[start] = elem_h.mean(dim=0)
            elif elem_h.size(0) <= n_codes:
                self.codebook[start:start + elem_h.size(0)] = elem_h
                if elem_h.size(0) < n_codes:
                    self.codebook[start + elem_h.size(0):end] = elem_h.mean(dim=0)
            else:
                km = KMeans(n_clusters=n_codes, n_init=1, max_iter=20)
                centers = torch.from_numpy(
                    km.fit(elem_h.cpu().numpy()).cluster_centers_
                ).to(elem_h)
                self.codebook[start:end] = centers

            self.ema_count[start:end] = 1.0
            self.ema_embed_sum[start:end] = self.codebook[start:end].clone()
        self.initialized.fill_(True)

    def forward(self, h, elem_indices):
        N = h.size(0)

        # Distance from each atom to every codebook entry
        all_dists = torch.cdist(h, self.codebook)

        # Mask so each atom can only match codes from its own element
        valid = torch.full((N, self.total_codes), float('inf'), device=h.device, dtype=h.dtype)
        for ei, (start, end) in enumerate(self._elem_ranges):
            mask = (elem_indices == ei)
            if mask.any():
                valid[mask, start:end] = all_dists[mask, start:end]

        # Snap each atom to its nearest valid code
        atom_types = valid.argmin(dim=1)
        z_q = self.codebook[atom_types]

        # Straight-through: forward uses z_q, backward passes gradient to h
        h_q = h + (z_q - h).detach()

        # Commitment loss (only for multi-code elements)
        multi_code_mask = torch.zeros(N, dtype=torch.bool, device=h.device)
        for ei, (start, end) in enumerate(self._elem_ranges):
            if end - start > 1:
                multi_code_mask |= (elem_indices == ei)

        if multi_code_mask.any():
            commitment_loss = ((h[multi_code_mask] - z_q[multi_code_mask].detach()) ** 2).mean()
        else:
            commitment_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        vq_loss = self.commitment_weight * commitment_loss

        # EMA codebook updates (training only)
        if self.training:
            with torch.no_grad():
                for ei, (start, end) in enumerate(self._elem_ranges):
                    if end - start <= 1:
                        continue
                    mask = (elem_indices == ei)
                    if not mask.any():
                        continue
                    local_idx = atom_types[mask] - start
                    self._ema_update(h[mask].detach(), local_idx, start, end)

        return h_q, atom_types, vq_loss

    @torch.no_grad()
    def _ema_update(self, elem_h, assignments, start, end):
        gamma = self.ema_decay
        n_codes = end - start
        onehot = torch.zeros(elem_h.size(0), n_codes, device=elem_h.device, dtype=elem_h.dtype)
        onehot.scatter_(1, assignments.unsqueeze(1), 1.0)

        self.ema_count[start:end] = gamma * self.ema_count[start:end] + (1 - gamma) * onehot.sum(0)
        self.ema_embed_sum[start:end] = gamma * self.ema_embed_sum[start:end] + (1 - gamma) * (onehot.T @ elem_h)

        count = self.ema_count[start:end].clamp(min=1e-5)
        self.codebook[start:end] = self.ema_embed_sum[start:end] / count.unsqueeze(1)

    @torch.no_grad()
    def get_active_codes(self):
        result = {}
        for elem in self.elements:
            start = self.elem_offset[elem]
            n_codes = self.n_codes_per_elem[elem]
            end = start + n_codes
            if n_codes <= 1:
                result[elem] = 1
                continue
            counts = self.ema_count[start:end]
            total = counts.sum()
            if total < 1e-8:
                result[elem] = n_codes
                continue
            probs = counts / total
            result[elem] = int((probs > 1.0 / (4 * n_codes)).sum().item())
        return result


class EnvironmentEncoder(nn.Module):

    def __init__(self, num_elements, embed_dim=16, num_rbf=8,
                 r_max_bohr=7.56, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(num_elements, embed_dim)
        self.embed.weight.data = self.embed.weight.data.double()

        # Gaussian RBF centers (fixed, evenly spaced)
        r_min_bohr = 0.5 * ANGSTROM_TO_BOHR
        centers = torch.linspace(r_min_bohr, r_max_bohr, num_rbf, dtype=torch.float64)
        width = (r_max_bohr - r_min_bohr) / num_rbf
        self.register_buffer('rbf_centers', centers)
        self.register_buffer('rbf_width', torch.tensor(width, dtype=torch.float64))

        # Per-layer distance filters: each layer gets its own RBF -> filter weights
        self.filter_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_rbf, embed_dim).double(),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim).double(),
            )
            for _ in range(num_layers)
        ])

    def forward(self, elem_idx, edge_index, distances):
        N = elem_idx.size(0)
        h = self.embed(elem_idx)

        # Gaussian RBF encoding (shared across layers)
        rbf = torch.exp(
            -0.5 * ((distances.unsqueeze(1) - self.rbf_centers) / self.rbf_width) ** 2
        )

        # Multi-layer message passing with residual connections
        src, tgt = edge_index
        for filter_net in self.filter_nets:
            W = filter_net(rbf)
            messages = h[tgt] * W
            aggregated = scatter_add(messages, src, dim=0, dim_size=N)
            h = h + aggregated

        return h


class MorseParameterPredictor(nn.Module):

    def __init__(self, embed_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.SiLU(),
            nn.Linear(embed_dim, 3).double(),
        )
        # Zero-init so GNN starts with zero corrections to base Morse params
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, edge_index):
        src, tgt = edge_index
        pair_feature = h[src] + h[tgt]
        return self.mlp(pair_feature)


class GNNMorseModel(nn.Module):

    def __init__(self, num_pairs, num_elements, elements, pair_names,
                 gnn_config, init_D_e=None, init_k=None, init_r0=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.elements = elements
        self.pair_names = pair_names

        self.vq_active = False

        # Default parameter initialization
        if init_D_e is None:
            init_D_e = np.full(num_pairs, 0.1 * EV_TO_HARTREE)
        if init_k is None:
            init_k = np.full(num_pairs, 2.0 * 0.1 * EV_TO_HARTREE * (1.5 * BOHR_TO_ANGSTROM) ** 2)
        if init_r0 is None:
            init_r0 = np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)

        # Pre-softplus values — softplus ensures D_e, k > 0
        self.raw_D_e = nn.Parameter(
            inverse_softplus(torch.tensor(init_D_e, dtype=torch.float64)))
        self.raw_k = nn.Parameter(
            inverse_softplus(torch.tensor(init_k, dtype=torch.float64)))
        self.raw_r0 = nn.Parameter(
            torch.tensor(init_r0, dtype=torch.float64))

        embed_dim = gnn_config.get('embed_dim', 16)
        num_rbf = gnn_config.get('num_rbf', 8)
        num_layers = gnn_config.get('num_layers', 2)
        r_max_bohr = gnn_config.get('r_max_angstroms', 4.0) * ANGSTROM_TO_BOHR
        self.env_encoder = EnvironmentEncoder(
            num_elements, embed_dim, num_rbf, r_max_bohr, num_layers)
        self.param_predictor = MorseParameterPredictor(embed_dim)

        self.vq = VectorQuantizer(
            elements=elements,
            embed_dim=embed_dim,
            max_codes_per_element=gnn_config.get('vq_max_codes', 4),
            core_elements=gnn_config.get('core_elements'),
            ema_decay=gnn_config.get('vq_ema_decay', 0.99),
            commitment_weight=gnn_config.get('vq_commitment_weight', 0.25),
        )
    #VQ-VAE has no decoder, subtypes are all learned through force loss
    def _compute_corrected_params(self, batch):
        raw_D = self.raw_D_e[batch.pair_indices]
        raw_k = self.raw_k[batch.pair_indices]
        raw_r0 = self.raw_r0[batch.pair_indices]
        vq_loss = torch.tensor(0.0, device=raw_D.device, dtype=raw_D.dtype)

        h = self.env_encoder(
            batch.element_indices, batch.edge_index, batch.distances)

        if self.vq_active:
            h, _, vq_loss = self.vq(h, batch.element_indices)

        corrections = self.param_predictor(h, batch.edge_index)
        D_e = nn.functional.softplus(raw_D + corrections[:, 0])
        k = nn.functional.softplus(raw_k + corrections[:, 1])
        r0 = raw_r0 + corrections[:, 2]

        # alpha = sqrt(k / (2*D_e))  from Morse spring constant k = 2*D_e*alpha^2
        alpha = torch.sqrt(k / (2.0 * D_e))

        return D_e, alpha, r0, vq_loss

    def forward(self, batch):
        D_e, alpha, r0, vq_loss = self._compute_corrected_params(batch)

        # Morse force: f(r) = 2*D_e*alpha*(exp(-2a(r-r0)) - exp(-a(r-r0)))
        x = batch.distances - r0
        exp_term = torch.exp(-alpha * x)
        scalar_force = 2.0 * D_e * alpha * (exp_term ** 2 - exp_term)

        # Accumulate forces on atoms
        force_vectors = -scalar_force.unsqueeze(1) * batch.edge_unit_vectors
        forces = scatter_add(
            force_vectors, batch.edge_index[0],
            dim=0, dim_size=batch.pos.size(0))

        return forces, vq_loss

    def get_per_edge_params(self, batch):
        with torch.no_grad():
            D_e, alpha, r0, _ = self._compute_corrected_params(batch)
        D_e_ev = D_e.cpu().numpy() * HARTREE_TO_EV
        alpha_inv_ang = alpha.cpu().numpy() / BOHR_TO_ANGSTROM
        r0_ang = r0.cpu().numpy() * BOHR_TO_ANGSTROM
        return {
            'D_e': D_e_ev,
            'alpha': alpha_inv_ang,
            'r0': r0_ang,
            'k': 2.0 * D_e_ev * alpha_inv_ang ** 2,
        }

    def get_base_params_display(self):
        D_e = nn.functional.softplus(self.raw_D_e).detach().cpu().numpy()
        k = nn.functional.softplus(self.raw_k).detach().cpu().numpy()
        r0 = self.raw_r0.detach().cpu().numpy()
        alpha = np.sqrt(k / (2.0 * D_e))
        return {
            'D_e': D_e * HARTREE_TO_EV,
            'alpha': alpha / BOHR_TO_ANGSTROM,
            'r0': r0 * BOHR_TO_ANGSTROM,
            'k': k * HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 2,
        }

    def get_atom_types(self, batch):
        if not self.vq_active:
            return None
        with torch.no_grad():
            h = self.env_encoder(
                batch.element_indices, batch.edge_index, batch.distances)
            _, atom_type_indices, _ = self.vq(h, batch.element_indices)

            type_names = []
            for i in range(batch.element_indices.size(0)):
                ei = batch.element_indices[i].item()
                elem = self.elements[ei]
                code_idx = atom_type_indices[i].item()
                local_code = code_idx - self.vq.elem_offset[elem]
                # Only add code suffix for multi-code elements
                if self.vq.n_codes_per_elem[elem] > 1:
                    type_names.append(f"{elem}_{local_code}")
                else:
                    type_names.append(elem)
            return atom_type_indices.cpu().numpy(), type_names

    def get_embeddings(self, batch):
        with torch.no_grad():
            return self.env_encoder(
                batch.element_indices, batch.edge_index, batch.distances
            ).cpu().numpy()

