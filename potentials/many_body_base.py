"""Abstract base class for many-body potentials (Tersoff, SW, etc.)."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

from .base import inverse_softplus


class ManyBodyPotential(ABC, nn.Module):
    """Base class for many-body potentials that use autograd for forces.

    Unlike BasePotential (which returns scalar_force per edge), many-body
    potentials compute total energy V(positions) and use
    torch.autograd.grad(-V, positions) to get forces. This handles
    angular/three-body terms that can't be decomposed into pairwise scalar forces.

    Parameters are stored as per-pair tensors of shape (num_pairs,), same as
    BasePotential, for consistency with the YAML config and display interface.
    """

    def __init__(self, num_pairs, initial_values=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.create_parameters(num_pairs, initial_values or {})

    # ── Flag ──────────────────────────────────────────────────────────

    @staticmethod
    def is_many_body():
        return True

    # ── Parameter interface (mirrors BasePotential) ───────────────────

    @staticmethod
    @abstractmethod
    def user_param_names():
        """User-facing parameter names for YAML config."""

    @staticmethod
    def param_kinds():
        """Return dict mapping user_param_name -> unit kind string."""
        return {}

    @staticmethod
    @abstractmethod
    def parameter_names():
        """Raw nn.Parameter attribute names stored on this module."""

    @abstractmethod
    def create_parameters(self, num_pairs, initial_values):
        """Create nn.Parameters from user-facing initial values dict."""

    @abstractmethod
    def get_constrained_params(self):
        """Return dict of user_param_name -> constrained tensor (num_pairs,)."""

    # ── Physics ───────────────────────────────────────────────────────

    @abstractmethod
    def compute_energy(self, positions, edge_index, element_indices,
                       pair_lookup_table, batch_vector, num_atoms):
        """Compute total potential energy as a differentiable scalar.

        Args:
            positions: (num_atoms, 3) positions in Bohr, requires_grad=True
            edge_index: (2, num_edges) source/target indices
            element_indices: (num_atoms,) element type per atom
            pair_lookup_table: (num_elements, num_elements) -> pair index
            batch_vector: (num_atoms,) batch assignment per atom
            num_atoms: total number of atoms

        Returns:
            scalar tensor: total energy in Hartree
        """

    # ── Triplet construction ──────────────────────────────────────────

    @staticmethod
    def build_triplets(edge_index, num_atoms):
        """Build triplet indices (i,j,k) from edge_index using CSR grouping.

        For each central atom i with neighbors j and k (j != k),
        produces a triplet (i, j, k) where the angle theta_jik is relevant.

        Args:
            edge_index: (2, num_edges) tensor [source, target]
            num_atoms: number of atoms

        Returns:
            triplet_i: (num_triplets,) central atom index
            triplet_j: (num_triplets,) first neighbor
            triplet_k: (num_triplets,) second neighbor
            edge_idx_ij: (num_triplets,) index into edge_index for i->j edge
            edge_idx_ik: (num_triplets,) index into edge_index for i->k edge
        """
        source, target = edge_index[0], edge_index[1]
        device = edge_index.device

        # Build CSR-like structure: for each atom i, find all edges where i is source
        # Sort edges by source atom
        sort_idx = torch.argsort(source)
        sorted_source = source[sort_idx]
        sorted_target = target[sort_idx]

        # Count neighbors per atom
        neighbor_count = torch.zeros(num_atoms, dtype=torch.long, device=device)
        neighbor_count.scatter_add_(0, sorted_source,
                                    torch.ones_like(sorted_source))

        # CSR row pointers
        row_ptr = torch.zeros(num_atoms + 1, dtype=torch.long, device=device)
        torch.cumsum(neighbor_count, dim=0, out=row_ptr[1:])

        # Build triplets
        triplet_i_list = []
        triplet_j_list = []
        triplet_k_list = []
        edge_idx_ij_list = []
        edge_idx_ik_list = []

        for i in range(num_atoms):
            start = row_ptr[i].item()
            end = row_ptr[i + 1].item()
            n_neighbors = end - start

            if n_neighbors < 2:
                continue

            # All pairs (j, k) where j != k among neighbors of i
            local_edges = sort_idx[start:end]  # original edge indices
            local_targets = sorted_target[start:end]  # neighbor atoms

            # Create all (j,k) pairs
            idx_j = torch.arange(n_neighbors, device=device)
            idx_k = torch.arange(n_neighbors, device=device)
            grid_j, grid_k = torch.meshgrid(idx_j, idx_k, indexing='ij')
            mask = grid_j != grid_k
            grid_j = grid_j[mask]
            grid_k = grid_k[mask]

            n_triplets = grid_j.shape[0]
            triplet_i_list.append(torch.full((n_triplets,), i,
                                             dtype=torch.long, device=device))
            triplet_j_list.append(local_targets[grid_j])
            triplet_k_list.append(local_targets[grid_k])
            edge_idx_ij_list.append(local_edges[grid_j])
            edge_idx_ik_list.append(local_edges[grid_k])

        if triplet_i_list:
            triplet_i = torch.cat(triplet_i_list)
            triplet_j = torch.cat(triplet_j_list)
            triplet_k = torch.cat(triplet_k_list)
            edge_idx_ij = torch.cat(edge_idx_ij_list)
            edge_idx_ik = torch.cat(edge_idx_ik_list)
        else:
            triplet_i = torch.empty(0, dtype=torch.long, device=device)
            triplet_j = torch.empty(0, dtype=torch.long, device=device)
            triplet_k = torch.empty(0, dtype=torch.long, device=device)
            edge_idx_ij = torch.empty(0, dtype=torch.long, device=device)
            edge_idx_ik = torch.empty(0, dtype=torch.long, device=device)

        return triplet_i, triplet_j, triplet_k, edge_idx_ij, edge_idx_ik

    # ── Equilibrium / RDF (many-body potentials don't use RDF init) ───

    @staticmethod
    def equilibrium_distance_param():
        """Many-body potentials don't have a simple equilibrium distance param."""
        return None

    @staticmethod
    def rdf_peak_to_param(peak_distance):
        return peak_distance

    # ── LAMMPS export ─────────────────────────────────────────────────

    @abstractmethod
    def write_lammps_file(self, pair_names, elements, cutoff_angstrom, filepath):
        """Write many-body potential file (.tersoff, .sw, etc.).

        Args:
            pair_names: list of 'El1-El2' strings
            elements: list of element strings
            cutoff_angstrom: cutoff in Angstroms
            filepath: base filepath (without extension)

        Returns:
            tuple: (pair_style_command, pair_coeff_lines, written_files)
        """

    # ── Display (mirrors BasePotential) ───────────────────────────────

    @abstractmethod
    def get_params_display(self):
        """Return dict of user_param_name -> np.array in physical units."""

    def display_header(self):
        names = self.user_param_names()
        return " | ".join(f"{n:>12}" for n in names)

    def display_row(self, pair_index):
        params = self.get_params_display()
        return " | ".join(
            f"{params[name][pair_index]:12.6f}" for name in self.user_param_names()
        )

    # ── Frozen mask ───────────────────────────────────────────────────

    def apply_frozen_mask(self, frozen_mask):
        if frozen_mask is None:
            return
        for pname in self.parameter_names():
            param = getattr(self, pname)
            if param.grad is not None:
                param.grad *= frozen_mask
