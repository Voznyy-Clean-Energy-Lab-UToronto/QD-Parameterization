import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from .utils import (
    HARTREE_TO_EV, BOHR_TO_ANGSTROM, EV_TO_HARTREE, ANGSTROM_TO_BOHR,
    inverse_softplus,
)


class EnvironmentEncoder(nn.Module):

    def __init__(self, num_elements, embed_dim=16, num_rbf=16, r_max_bohr=7.56):
        super().__init__()
        self.embed_dim = embed_dim

        # Element embedding
        self.embed = nn.Embedding(num_elements, embed_dim)
        # Cast to float64
        self.embed.weight.data = self.embed.weight.data.double()

        # Gaussian RBF centers (fixed, not learned)
        r_min_bohr = 0.5 * ANGSTROM_TO_BOHR   # 0.5 Ang
        centers = torch.linspace(r_min_bohr, r_max_bohr, num_rbf, dtype=torch.float64)
        width = (r_max_bohr - r_min_bohr) / num_rbf
        self.register_buffer('rbf_centers', centers)
        self.register_buffer('rbf_width', torch.tensor(width, dtype=torch.float64))

        # Continuous-filter MLP: RBF → filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, embed_dim).double(),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim).double(),
        )

    def forward(self, elem_idx, edge_index, distances):
        N = elem_idx.size(0)
        h = self.embed(elem_idx)  # (N, d)

        # Gaussian RBF encoding of distances
        rbf = torch.exp(
            -0.5 * ((distances.unsqueeze(1) - self.rbf_centers) / self.rbf_width) ** 2
        )  # (E, num_rbf)

        # Continuous filter convolution
        W = self.filter_net(rbf)  # (E, d)

        # Messages: neighbor embedding * distance-dependent filter
        src, tgt = edge_index
        messages = h[tgt] * W  # (E, d) — info FROM target TO source

        # Aggregate at source atoms
        agg = scatter_add(messages, src, dim=0, dim_size=N)  # (N, d)

        # Residual update
        return h + agg


class MorseParameterPredictor(nn.Module):

    def __init__(self, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.SiLU(),
            nn.Linear(embed_dim, 3).double(),
        )
        # Zero-init output layer → corrections start at zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, edge_index):
        src, tgt = edge_index
        # Symmetric pair feature: same for (i,j) and (j,i)
        pair_feat = h[src] + h[tgt]  # (E, d)
        return self.mlp(pair_feat)


class GNNMorseModel(nn.Module):
    def __init__(self, num_pairs, num_elements, elements, pair_names,
                 gnn_config, init_D_e=None, init_alpha=None, init_r0=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.elements = elements
        self.pair_names = pair_names
        self.use_gnn = gnn_config.get('enabled', True)

        # Base Morse parameters (per pair type, internal units)
        if init_D_e is None:
            init_D_e = np.full(num_pairs, 0.1 * EV_TO_HARTREE)
        if init_alpha is None:
            init_alpha = np.full(num_pairs, 1.5 * BOHR_TO_ANGSTROM)
        if init_r0 is None:
            init_r0 = np.full(num_pairs, 3.0 * ANGSTROM_TO_BOHR)

        self.raw_D_e = nn.Parameter(
            inverse_softplus(torch.tensor(init_D_e, dtype=torch.float64)))
        self.raw_alpha = nn.Parameter(
            inverse_softplus(torch.tensor(init_alpha, dtype=torch.float64)))
        self.raw_r0 = nn.Parameter(
            torch.tensor(init_r0, dtype=torch.float64))

        # GNN submodules
        if self.use_gnn:
            embed_dim = gnn_config.get('embed_dim', 16)
            num_rbf = gnn_config.get('num_rbf', 16)
            r_max_bohr = gnn_config.get('r_max_angstroms', 4.0) * ANGSTROM_TO_BOHR
            self.env_encoder = EnvironmentEncoder(
                num_elements, embed_dim, num_rbf, r_max_bohr)
            self.param_predictor = MorseParameterPredictor(embed_dim)

        # Learnable energy offset
        self.energy_offset = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64))

    def _compute_corrected_params(self, batch):
        raw_D = self.raw_D_e[batch.pair_indices]
        raw_a = self.raw_alpha[batch.pair_indices]
        raw_r0 = self.raw_r0[batch.pair_indices]

        if self.use_gnn:
            h = self.env_encoder(
                batch.element_indices, batch.edge_index, batch.distances)
            corr = self.param_predictor(h, batch.edge_index)
            D_e = nn.functional.softplus(raw_D + corr[:, 0])
            alpha = nn.functional.softplus(raw_a + corr[:, 1])
            r0 = raw_r0 + corr[:, 2]
        else:
            D_e = nn.functional.softplus(raw_D)
            alpha = nn.functional.softplus(raw_a)
            r0 = raw_r0
        return D_e, alpha, r0

    def forward(self, batch):
        D_e, alpha, r0 = self._compute_corrected_params(batch)

        # Morse potential
        x = batch.distances - r0
        exp1 = torch.exp(-alpha * x)
        exp2 = exp1 * exp1
        morse_V = D_e * (1.0 - exp1) ** 2
        morse_sf = 2.0 * D_e * alpha * (exp2 - exp1)  # -dV/dr

        # Force accumulation
        force_vectors = -morse_sf.unsqueeze(1) * batch.edge_unit_vectors
        forces = scatter_add(
            force_vectors, batch.edge_index[0],
            dim=0, dim_size=batch.pos.size(0))

        # Energy per graph (divide by 2 for double-counted edges)
        if hasattr(batch, 'batch') and batch.batch is not None:
            edge_batch = batch.batch[batch.edge_index[0]]
            num_graphs = int(batch.batch.max().item()) + 1
        else:
            edge_batch = torch.zeros(
                batch.edge_index.size(1), dtype=torch.long,
                device=batch.pos.device)
            num_graphs = 1

        graph_energies = scatter_add(
            morse_V, edge_batch, dim=0, dim_size=num_graphs) / 2.0
        graph_energies = graph_energies + self.energy_offset

        return forces, graph_energies

    def get_per_edge_params(self, batch):
        with torch.no_grad():
            D_e, alpha, r0 = self._compute_corrected_params(batch)
        return {
            'D_e': (D_e.cpu().numpy() * HARTREE_TO_EV),
            'alpha': (alpha.cpu().numpy() / BOHR_TO_ANGSTROM),
            'r0': (r0.cpu().numpy() * BOHR_TO_ANGSTROM),
        }

    def get_base_params_display(self):
        D_e = nn.functional.softplus(self.raw_D_e).detach().cpu().numpy()
        alpha = nn.functional.softplus(self.raw_alpha).detach().cpu().numpy()
        r0 = self.raw_r0.detach().cpu().numpy()
        return {
            'D_e': D_e * HARTREE_TO_EV,
            'alpha': alpha / BOHR_TO_ANGSTROM,
            'r0': r0 * BOHR_TO_ANGSTROM,
        }

    def get_embeddings(self, batch):
        with torch.no_grad():
            if self.use_gnn:
                return self.env_encoder(
                    batch.element_indices, batch.edge_index, batch.distances
                ).cpu().numpy()
            else:
                return None

    def apply_frozen_mask(self, frozen_mask):
        for param in [self.raw_D_e, self.raw_alpha, self.raw_r0]:
            if param.grad is not None:
                param.grad *= frozen_mask
