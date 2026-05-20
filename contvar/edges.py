import numpy as np
import torch


class SaladStyleEdgeBuilder:
    """
    Salad-style edge construction for protein graphs.

    Combines three types of neighbors:
    1. Index neighbors: Based on residue sequence distance
    2. Spatial neighbors: K-nearest neighbors by Euclidean distance (Ca)
    3. Random neighbors: Probabilistically sampled with probability ~ 1/d^3

    This creates a sparse but information-rich edge set that captures both
    local sequence context and 3D structural relationships.
    """

    def __init__(self, num_index=16, num_spatial=16, num_random=16):
        self.num_index = num_index
        self.num_spatial = num_spatial
        self.num_random = num_random
        self.total_neighbors = num_index + num_spatial + num_random

    def extract_neighbors(self, coords, residue_indices, chain_ids=None):
        """
        Extract neighbors for each residue using hybrid approach.

        Args:
            coords: (N, 3) array of Ca coordinates
            residue_indices: (N,) array of residue indices
            chain_ids: (N,) array of chain identifiers (optional)

        Returns:
            neighbors: (N, K) array of neighbor indices (-1 for invalid)
            distances: (N, K) array of Euclidean distances
            neighbor_types: (N, K) array of neighbor type (0=index, 1=spatial, 2=random)
        """
        N = len(coords)
        coords = np.array(coords)
        residue_indices = np.array(residue_indices)

        if chain_ids is None:
            chain_ids = np.zeros(N, dtype=np.int32)
        else:
            chain_ids = np.array(chain_ids)

        # Compute pairwise distance matrix
        dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)

        # Mask for same chain (only connect within same chain by default)
        same_chain = chain_ids[:, None] == chain_ids[None, :]

        # Initialize neighbor tracking
        all_neighbors = []
        all_distances = []
        all_types = []

        # 1. Index-based neighbors (sequence proximity)
        index_neighbors, index_dists = self._get_index_neighbors(
            residue_indices, chain_ids, dist_matrix, same_chain
        )
        all_neighbors.append(index_neighbors)
        all_distances.append(index_dists)
        all_types.append(np.zeros_like(index_neighbors))  # type 0 = index

        # Create mask for already selected neighbors
        selected = set()
        for i in range(N):
            for j in index_neighbors[i]:
                if j >= 0:
                    selected.add((i, j))

        # 2. Spatial neighbors (k-nearest by Euclidean distance)
        spatial_neighbors, spatial_dists = self._get_spatial_neighbors(
            dist_matrix, same_chain, selected
        )
        all_neighbors.append(spatial_neighbors)
        all_distances.append(spatial_dists)
        all_types.append(np.ones_like(spatial_neighbors))  # type 1 = spatial

        # Update selected
        for i in range(N):
            for j in spatial_neighbors[i]:
                if j >= 0:
                    selected.add((i, j))

        # 3. Random neighbors (probability ~ 1/d^3, Gumbel top-k trick)
        random_neighbors, random_dists = self._get_random_neighbors(
            dist_matrix, same_chain, selected
        )
        all_neighbors.append(random_neighbors)
        all_distances.append(random_dists)
        all_types.append(2 * np.ones_like(random_neighbors))  # type 2 = random

        # Concatenate all neighbor types
        neighbors = np.concatenate(all_neighbors, axis=1)
        distances = np.concatenate(all_distances, axis=1)
        neighbor_types = np.concatenate(all_types, axis=1)

        return neighbors, distances, neighbor_types

    def _get_index_neighbors(self, residue_indices, chain_ids, dist_matrix, same_chain):
        """Get neighbors based on residue sequence index distance."""
        N = len(residue_indices)

        # Compute index distance
        index_dist = np.abs(residue_indices[:, None] - residue_indices[None, :])

        # Mask out different chains
        index_dist = np.where(same_chain, index_dist, np.inf)

        # Self-distance is infinity (don't select self)
        np.fill_diagonal(index_dist, np.inf)

        # Get k-nearest by index distance
        neighbors = np.zeros((N, self.num_index), dtype=np.int32)
        distances = np.zeros((N, self.num_index), dtype=np.float32)

        for i in range(N):
            sorted_idx = np.argsort(index_dist[i])[:self.num_index]
            valid = index_dist[i, sorted_idx] < np.inf
            neighbors[i, valid] = sorted_idx[valid]
            neighbors[i, ~valid] = -1
            distances[i] = np.where(valid, dist_matrix[i, sorted_idx], 0)

        return neighbors, distances

    def _get_spatial_neighbors(self, dist_matrix, same_chain, selected):
        """Get k-nearest neighbors by Euclidean distance."""
        N = dist_matrix.shape[0]

        # Create working copy with masking
        spatial_dist = dist_matrix.copy()
        np.fill_diagonal(spatial_dist, np.inf)
        spatial_dist = np.where(same_chain, spatial_dist, np.inf)

        # Mask already selected neighbors
        for (i, j) in selected:
            spatial_dist[i, j] = np.inf

        neighbors = np.zeros((N, self.num_spatial), dtype=np.int32)
        distances = np.zeros((N, self.num_spatial), dtype=np.float32)

        for i in range(N):
            sorted_idx = np.argsort(spatial_dist[i])[:self.num_spatial]
            valid = spatial_dist[i, sorted_idx] < np.inf
            neighbors[i, valid] = sorted_idx[valid]
            neighbors[i, ~valid] = -1
            distances[i] = np.where(valid, dist_matrix[i, sorted_idx], 0)

        return neighbors, distances

    def _get_random_neighbors(self, dist_matrix, same_chain, selected):
        """
        Get random neighbors with probability proportional to 1/d^3.
        Uses the Gumbel top-k trick for efficient sampling.
        """
        N = dist_matrix.shape[0]

        # Weight = -3 * log(distance + eps) -> higher weight for closer residues
        eps = 1e-6
        weight = -3 * np.log(dist_matrix + eps)

        # Apply Gumbel noise for randomization
        uniform = np.random.uniform(eps, 1 - eps, weight.shape)
        gumbel = np.log(-np.log(uniform))

        # Perturbed weights
        random_dist = -(weight - gumbel)

        # Mask
        np.fill_diagonal(random_dist, np.inf)
        random_dist = np.where(same_chain, random_dist, np.inf)

        # Mask already selected
        for (i, j) in selected:
            random_dist[i, j] = np.inf

        neighbors = np.zeros((N, self.num_random), dtype=np.int32)
        distances = np.zeros((N, self.num_random), dtype=np.float32)

        for i in range(N):
            sorted_idx = np.argsort(random_dist[i])[:self.num_random]
            valid = random_dist[i, sorted_idx] < np.inf
            neighbors[i, valid] = sorted_idx[valid]
            neighbors[i, ~valid] = -1
            distances[i] = np.where(valid, dist_matrix[i, sorted_idx], 0)

        return neighbors, distances

    def build_edge_index_and_attr(self, coords, residue_indices, chain_ids=None,
                                   use_rbf=True, num_rbf=16, d_max=22.0):
        """
        Build edge_index and edge_attr tensors for PyG.

        Returns:
            edge_index: (2, E) tensor
            edge_attr: (E, D) tensor with features:
                - RBF distance encoding (num_rbf dims)
                - Neighbor type one-hot (3 dims: index, spatial, random)
                - Relative sequence distance (1 dim, normalized)
        """
        neighbors, distances, neighbor_types = self.extract_neighbors(
            coords, residue_indices, chain_ids
        )

        N = len(coords)
        edge_list = []
        edge_attr_list = []

        residue_indices = np.array(residue_indices)

        for i in range(N):
            for k in range(neighbors.shape[1]):
                j = neighbors[i, k]
                if j < 0:  # Invalid neighbor
                    continue

                dist = distances[i, k]
                ntype = int(neighbor_types[i, k])

                # Distance encoding
                if use_rbf:
                    dist_feat = self._distance_rbf(dist, d_max=d_max, num_bins=num_rbf)
                else:
                    dist_feat = [dist]

                # Neighbor type one-hot
                type_feat = [0.0, 0.0, 0.0]
                type_feat[ntype] = 1.0

                # Relative sequence distance (normalized)
                seq_dist = abs(residue_indices[i] - residue_indices[j]) / max(N, 1)

                edge_list.append([i, j])
                edge_attr_list.append(dist_feat + type_feat + [seq_dist])

        if len(edge_list) == 0:
            return (torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, num_rbf + 4), dtype=torch.float))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        return edge_index, edge_attr

    def _distance_rbf(self, distance, d_min=0.0, d_max=22.0, num_bins=16):
        """Compute radial basis function encoding of distance."""
        step = (d_max - d_min) / num_bins
        centers = d_min + np.arange(num_bins) * step + step / 2
        rbf = np.exp(-((distance - centers) / step) ** 2)
        return rbf.tolist()

    @property
    def edge_attr_dim(self):
        """Return the dimension of edge attributes."""
        # RBF (16) + neighbor_type (3) + seq_dist (1) = 20
        return 20
