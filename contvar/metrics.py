import numpy as np
import torch
import torch.nn.functional as F


def compute_detailed_metrics(anchor_embed, pos_embed, neg_embed):
    """Compute retrieval, alignment and uniformity metrics."""
    metrics = {}
    batch_size = anchor_embed.size(0)

    # ALIGNMENT & UNIFORMITY
    align_loss = (anchor_embed - pos_embed).norm(p=2, dim=1).pow(2).mean()
    metrics["Alignment"] = align_loss.item()

    all_embeds = torch.cat([anchor_embed, pos_embed, neg_embed], dim=0)
    if len(all_embeds) > 1:
        dist_sq = torch.pdist(all_embeds, p=2).pow(2)
        unif_loss = dist_sq.mul(-2).exp().mean().log()
        metrics["Uniformity"] = unif_loss.item()
    else:
        metrics["Uniformity"] = 0.0

    # RETRIEVAL METRICS
    candidates = torch.cat([pos_embed, neg_embed], dim=0)
    dists = torch.cdist(anchor_embed, candidates, p=2)

    total_mrr = 0
    for i in range(batch_size):
        target_idx = i
        sorted_indices = torch.argsort(dists[i], descending=False)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
        total_mrr += 1.0 / rank

    metrics["MRR"] = total_mrr / batch_size

    return metrics


def compute_embedding_stats(model, loader, device, criterion, max_batches=None):
    """
    Compute global and local embedding statistics over batches (cosine similarity, std).
    """
    model.eval()
    stats = {
        "global_cos_sim_anchor_pos": [],
        "global_cos_sim_anchor_neg": [],
        "global_embedding_std": [],
        "local_cos_sim_anchor_pos": [],
        "local_cos_sim_anchor_neg": [],
        "local_embedding_std": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch is None:
                continue
            ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives = batch
            ba = ba.to(device)
            bp = bp.to(device)
            bn = bn.to(device)
            neg_counts = neg_counts.to(device)
            mut_pos_positive = mut_pos_positive.to(device)
            mut_pos_negatives = mut_pos_negatives.to(device)
            ea_g, _ = model(ba)
            ep_g, ep_l = model(bp, mut_pos=mut_pos_positive)
            en_g, en_l = model(bn, mut_pos=mut_pos_negatives)
            _, _, en_neg_g, mining_stats = criterion(ea_g, ep_g, en_g, neg_counts)
            hardest_indices = mining_stats["hardest_indices"]
            cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
            flat_idx = cumsum + hardest_indices
            mut_pos_neg_selected = mut_pos_negatives[flat_idx]
            _, la_at_pos = model(ba, mut_pos=mut_pos_positive)
            _, la_at_neg = model(ba, mut_pos=mut_pos_neg_selected)
            zn_l_selected = en_l[flat_idx]
            stats["global_cos_sim_anchor_pos"].append(F.cosine_similarity(ea_g, ep_g, dim=1).mean().item())
            stats["global_cos_sim_anchor_neg"].append(F.cosine_similarity(ea_g, en_neg_g, dim=1).mean().item())
            stats["local_cos_sim_anchor_pos"].append(F.cosine_similarity(la_at_pos, ep_l, dim=1).mean().item())
            stats["local_cos_sim_anchor_neg"].append(F.cosine_similarity(la_at_neg, zn_l_selected, dim=1).mean().item())
            all_g = torch.cat([ea_g, ep_g, en_neg_g], dim=0)
            all_l = torch.cat([la_at_pos, ep_l, zn_l_selected], dim=0)
            stats["global_embedding_std"].append(all_g.std(dim=0).mean().item())
            stats["local_embedding_std"].append(all_l.std(dim=0).mean().item())
    out = {}
    for k, v in stats.items():
        out[k] = np.mean(v) if v else 0.0
    return out
