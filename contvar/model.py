import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class DeepProteinGAT(nn.Module):
    """Two-layer GATv2 model with residual connections, edge features,
    and clean projection heads for metric learning.

    This model supports both edge construction modes:
    - SALAD-style edges: RBF distance encoding + neighbor type + sequence distance (edge_dim=20)
    - Graphein edges: kNN Euclidean distance only (edge_dim=1)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4,
                 edge_dim=20, projection_hidden_dim=None):
        super().__init__()

        self.edge_dim = edge_dim
        conv_out_dim = hidden_dim * heads

        if projection_hidden_dim is None:
            projection_hidden_dim = output_dim * 2

        # Edge feature embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GATv2 layer 1: input_dim -> hidden_dim * heads
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True,
                               dropout=0.0, edge_dim=hidden_dim)
        self.norm1 = nn.LayerNorm(conv_out_dim)

        # GATv2 layer 2: conv_out_dim -> hidden_dim * heads
        self.conv2 = GATv2Conv(conv_out_dim, hidden_dim, heads=heads, concat=True,
                               dropout=0.0, edge_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(conv_out_dim)

        # Residual projection (match input_dim -> conv_out_dim)
        self.input_proj = nn.Linear(input_dim, conv_out_dim)

        # Global projection head
        self.projection = nn.Sequential(
            nn.Linear(conv_out_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, output_dim),
        )
        # Local projection head
        self.projection_local = nn.Sequential(
            nn.Linear(conv_out_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, output_dim),
        )

        # Optional GO heads for phase-0 semantic pretraining
        # They operate on the global embedding space and are shared across phases.
        self.head_mf = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        self.head_bp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        self.head_cc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _gnn_forward(self, data):
        """Run GNN backbone, return node features and metadata for reuse."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        x = x.float()

        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.float()
            edge_embed = self.edge_encoder(edge_attr)
        else:
            edge_embed = None

        # Residual: project original ESM2 features to match conv output dim
        x_residual = self.input_proj(x)

        # GATv2 layer 1 with residual connection
        # Single GATv2 layer with residual connection
        x = self.conv1(x, edge_index, edge_attr=edge_embed)
        x = self.norm1(x)
        x = x + x_residual
        x = F.elu(x)

        # GATv2 layer 2 with residual connection
        x_residual2 = x
        x = self.conv2(x, edge_index, edge_attr=edge_embed)
        x = self.norm2(x)
        x = x + x_residual2
        x = F.elu(x)

        res_num = data.residue_number.to(x.device) if hasattr(data, 'residue_number') and data.residue_number is not None else None
        return x, batch, res_num

    def _extract_local(self, x, batch, res_num, mut_pos):
        """Extract local embeddings at given positions from pre-computed node features.
        This is used during mining to efficiently get local embeddings without re-running the GNN."""
        B = batch.max().item() + 1
        device = x.device
        mut_pos = mut_pos.to(device)
        x_local = torch.zeros(B, x.size(1), device=device, dtype=x.dtype)
        for i in range(B):
            mask = (batch == i) & (res_num == mut_pos[i])
            if mask.any() and mut_pos[i] >= 0:
                idx = mask.nonzero(as_tuple=True)[0][0]
                x_local[i] = x[idx]
            else:
                graph_mask = (batch == i)
                x_local[i] = x[graph_mask].mean(0)
        x_local = self.projection_local(x_local)
        x_local = F.normalize(x_local, p=2, dim=1)
        return x_local

    def forward(self, data, mut_pos=None):
        x, batch, res_num = self._gnn_forward(data)

        # Global embedding
        x_global = global_mean_pool(x, batch)
        x_global = self.projection(x_global)
        x_global = F.normalize(x_global, p=2, dim=1)

        # Local: embedding at mut_pos per graph (fallback to graph mean if position missing)
        if mut_pos is not None and res_num is not None:
            x_local = self._extract_local(x, batch, res_num, mut_pos)
        else:
            x_local = x_global

        return x_global, x_local

    def forward_with_nodes(self, data, mut_pos=None):
        """Forward pass that also returns node-level features for reuse.

        Returns (x_global, x_local, node_ctx) where node_ctx = (x, batch, res_num)
        can be passed to _extract_local for cheap additional position extractions.
        """
        x, batch, res_num = self._gnn_forward(data)

        x_global = global_mean_pool(x, batch)
        x_global = self.projection(x_global)
        x_global = F.normalize(x_global, p=2, dim=1)

        if mut_pos is not None and res_num is not None:
            x_local = self._extract_local(x, batch, res_num, mut_pos)
        else:
            x_local = x_global

        return x_global, x_local, (x, batch, res_num)

    # ------------------------------------------------------------------
    # GO semantic pretraining helpers (phase 0)
    # ------------------------------------------------------------------
    def forward_go_head(self, data, ontology: str):
        """
        Compute ontology-specific embedding for GO semantic pretraining.

        Args:
            data: PyG Data batch
            ontology: one of {'mf', 'bp', 'cc'}
        """
        z_g, _ = self.forward(data, mut_pos=None)

        if ontology == "mf":
            z = self.head_mf(z_g)
        elif ontology == "bp":
            z = self.head_bp(z_g)
        elif ontology == "cc":
            z = self.head_cc(z_g)
        else:
            raise ValueError(f"Unknown ontology: {ontology}")

        z = F.normalize(z, p=2, dim=1)
        return z
