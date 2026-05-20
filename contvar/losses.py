import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiHardMiningTripletLoss(nn.Module):
    """
    Triplet loss with hard and semi-hard negative mining.

    Hard negatives: d(anchor, negative) < d(anchor, positive)
    Semi-hard negatives: d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin

    Averages loss over all qualifying negatives. Falls back to standard triplet loss
    if no hard/semi-hard negatives are found.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, neg_embeddings, neg_counts):
        """
        Fully vectorized forward pass for semi-hard/hard mining.

        Args:
            anchor: [batch_size, embed_dim]
            positive: [batch_size, embed_dim]
            neg_embeddings: [total_negatives, embed_dim]
            neg_counts: [batch_size]

        Returns:
            loss, avg_neg_dist, representative_neg, mining_stats
        """
        batch_size = anchor.size(0)
        embed_dim = anchor.size(1)
        device = anchor.device

        # Compute positive distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)

        # Build sample indices for each negative
        sample_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            neg_counts
        )

        # Expand anchor to match each negative
        anchor_expanded = anchor[sample_indices]

        # Compute all negative distances vectorized
        neg_dists_flat = F.pairwise_distance(anchor_expanded, neg_embeddings, p=2)

        # Expand pos_dist to match each negative
        pos_dist_expanded = pos_dist[sample_indices]

        # Identify hard negatives: d(a,n) < d(a,p)
        hard_mask = neg_dists_flat < pos_dist_expanded

        # Identify semi-hard negatives: d(a,p) <= d(a,n) < d(a,p) + margin
        semi_hard_mask = (neg_dists_flat >= pos_dist_expanded) & \
                         (neg_dists_flat < pos_dist_expanded + self.margin)

        # Combined mask for qualifying negatives
        qualifying_mask = hard_mask | semi_hard_mask

        # Mining statistics
        hard_count = hard_mask.sum().item()
        semi_hard_count = semi_hard_mask.sum().item()
        total_qualifying = qualifying_mask.sum().item()
        total_negatives = len(neg_embeddings)

        mining_stats = {
            "hard_count": hard_count,
            "semi_hard_count": semi_hard_count,
            "total_qualifying": total_qualifying,
            "total_negatives": total_negatives,
        }

        # Compute triplet losses for ALL negatives
        all_losses = F.relu(pos_dist_expanded - neg_dists_flat + self.margin)

        num_qualifying = total_qualifying

        # Pre-compute padding structure
        max_negs = neg_counts.max().item()
        neg_dists_padded = torch.full((batch_size, max_negs), float('inf'), device=device)
        valid_neg_mask = (
            torch.arange(max_negs, device=device).unsqueeze(0) < neg_counts.unsqueeze(1)
        )

        # Build position indices within each sample
        cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
        positions = torch.arange(len(sample_indices), device=device) - cumsum[sample_indices]

        # Scatter distances into padded structure
        neg_dists_padded[sample_indices, positions] = neg_dists_flat

        if num_qualifying > 0:
            loss = (all_losses * qualifying_mask.float()).sum() / num_qualifying

            qualifying_padded = torch.zeros((batch_size, max_negs), dtype=torch.bool, device=device)
            qualifying_padded[sample_indices, positions] = qualifying_mask

            qualifying_dists = torch.where(qualifying_padded, neg_dists_padded,
                                           torch.tensor(float('inf'), device=device))
            qualifying_counts = qualifying_padded.sum(dim=1)
            safe_qualifying_dists = torch.where(
                qualifying_padded, neg_dists_padded, torch.zeros_like(neg_dists_padded)
            )
            avg_neg_dist = safe_qualifying_dists.sum(dim=1) / qualifying_counts.clamp(min=1)

            no_qualifying = qualifying_counts == 0
            if no_qualifying.any():
                valid_neg_dists = torch.where(
                    valid_neg_mask, neg_dists_padded, torch.tensor(float('inf'), device=device)
                )
                min_dists, min_indices = valid_neg_dists.min(dim=1)
                avg_neg_dist = torch.where(no_qualifying, min_dists, avg_neg_dist)
            else:
                min_indices = None

            hardest_indices = qualifying_dists.argmin(dim=1)
            if min_indices is not None:
                hardest_indices = torch.where(no_qualifying, min_indices, hardest_indices)

        else:
            valid_neg_dists = torch.where(
                valid_neg_mask, neg_dists_padded, torch.tensor(float('inf'), device=device)
            )
            min_neg_dist, hardest_indices = valid_neg_dists.min(dim=1)
            loss = F.relu(pos_dist - min_neg_dist + self.margin).mean()
            avg_neg_dist = min_neg_dist

        mining_stats["hardest_indices"] = hardest_indices

        # Extract representative negative embeddings
        neg_embeddings_padded = torch.zeros((batch_size, max_negs, embed_dim), device=device)
        neg_embeddings_padded[sample_indices, positions] = neg_embeddings

        representative_neg = neg_embeddings_padded[
            torch.arange(batch_size, device=device),
            hardest_indices
        ]

        return loss, avg_neg_dist, representative_neg, mining_stats


class StandardTripletLoss(nn.Module):
    """Wrapper for PyTorch's TripletMarginLoss alternative when no hard/semi-hard mining is desired."""

    def __init__(self, margin=0.3, p=2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p, reduction='mean')

    def forward(self, anchor, positive, neg_embeddings, neg_counts):
        batch_size = anchor.size(0)
        device = anchor.device

        # Extract first negative for each sample
        cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
        first_neg_indices = cumsum

        negative = neg_embeddings[first_neg_indices]

        loss = self.triplet_loss(anchor, positive, negative)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        mining_stats = {
            "hard_count": 0,
            "semi_hard_count": 0,
            "total_qualifying": 0,
            "total_negatives": len(neg_embeddings),
            "hardest_indices": torch.zeros(batch_size, dtype=torch.long, device=device)
        }

        return loss, neg_dist, negative, mining_stats


def get_loss_function(config):
    """Factory function to get the appropriate loss function based on configuration."""
    loss_type = config.loss_type.lower()

    if loss_type == "semi_hard":
        print(f"Using SemiHardMiningTripletLoss with margin={config.margin}")
        return SemiHardMiningTripletLoss(margin=config.margin)

    elif loss_type == "standard":
        print(f"Using StandardTripletLoss (nn.TripletMarginLoss) with margin={config.margin}")
        return StandardTripletLoss(margin=config.margin, p=2)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: 'semi_hard', 'standard'")
