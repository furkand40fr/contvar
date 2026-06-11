import os
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from contvar.data.collate import parse_mut_pos_from_path


def load_processed_graph_static(path, processed_dir):
    """Load a processed protein graph (.pt) from disk. Standalone, no Dataset dependency."""
    pdb_code = os.path.splitext(os.path.basename(path))[0]
    pt_path = os.path.join(processed_dir, f"{pdb_code}.pt")
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    else:
        print(f"Warning: processed file not found: {pt_path}")
        return None


def mine_negatives_streaming(model, anchor_embeds, pos_dists, batch_triplets,
                              processed_dir, device, margin, max_negatives, chunk_size):
    """
    Evaluate ALL negatives for each protein family in memory-efficient chunks,
    keeping only hard/semi-hard negatives (up to max_negatives per family).

    Args:
        model: GNN model (set to eval mode internally)
        anchor_embeds: [batch_size, embed_dim] pre-computed anchor embeddings
        pos_dists: [batch_size] distances to positives
        batch_triplets: list of triplet dicts with 'negatives' paths
        processed_dir: path to processed/ directory with .pt files
        device: torch device
        margin: triplet margin for semi-hard boundary
        max_negatives: max negatives to keep per family
        chunk_size: number of negatives to GPU-forward at once

    Returns:
        all_qualifying_negs, all_qualifying_mut_pos, neg_count_list, mining_info
    """
    batch_size = len(batch_triplets)
    all_qualifying_negs = []
    all_qualifying_mut_pos = []
    neg_count_list = []

    total_evaluated = 0
    total_hard = 0
    total_semi_hard = 0

    model.eval()

    for i in range(batch_size):
        anchor_embed_i = anchor_embeds[i:i+1]
        pos_dist_i = pos_dists[i].item()

        neg_paths = batch_triplets[i]['negatives']

        family_qualifying = []
        closest_neg = None
        closest_dist = float('inf')

        for chunk_start in range(0, len(neg_paths), chunk_size):
            chunk_paths = neg_paths[chunk_start:chunk_start + chunk_size]
            chunk_data = []
            chunk_mut_pos = []

            for p in chunk_paths:
                data = load_processed_graph_static(p, processed_dir)
                if data is not None:
                    chunk_data.append(data)
                    mut_p = parse_mut_pos_from_path(p)
                    chunk_mut_pos.append(mut_p if mut_p is not None else -1)

            if not chunk_data:
                continue

            chunk_batch = Batch.from_data_list(chunk_data).to(device)
            chunk_mut_pos = torch.tensor(
                chunk_mut_pos, dtype=torch.long, device=device
            )

            with torch.no_grad():
                chunk_embed, _ = model(chunk_batch, mut_pos=chunk_mut_pos)

            anchor_expanded = anchor_embed_i.expand(len(chunk_data), -1)
            chunk_dists = F.pairwise_distance(anchor_expanded, chunk_embed, p=2)

            for j in range(len(chunk_data)):
                d = chunk_dists[j].item()
                total_evaluated += 1

                mut_p_val = int(chunk_mut_pos[j].item())

                if d < closest_dist:
                    closest_dist = d
                    closest_neg = (chunk_data[j], d, mut_p_val)

                is_hard = d < pos_dist_i
                is_semi = (d >= pos_dist_i) and (d < pos_dist_i + margin)

                if is_hard:
                    family_qualifying.append((chunk_data[j], d, mut_p_val))
                    total_hard += 1
                elif is_semi:
                    family_qualifying.append((chunk_data[j], d, mut_p_val))
                    total_semi_hard += 1

            del chunk_batch, chunk_embed, chunk_dists

        # Select random from qualifying negatives (not hardeest)
        if family_qualifying:
            if len(family_qualifying) <= max_negatives:
                selected = family_qualifying
            else:
                selected = random.sample(family_qualifying, max_negatives)
        else:
            if closest_neg is not None:
                selected = [closest_neg]
            else:
                selected = []

        for data_obj, dist, mut_p in selected:
            all_qualifying_negs.append(data_obj)
            all_qualifying_mut_pos.append(mut_p)
        neg_count_list.append(len(selected))

    mining_info = {
        "total_evaluated": total_evaluated,
        "total_qualifying": total_hard + total_semi_hard,
        "streaming_hard": total_hard,
        "streaming_semi_hard": total_semi_hard,
    }

    return all_qualifying_negs, all_qualifying_mut_pos, neg_count_list, mining_info


def streaming_mining_batch_iterator(model, triplets, processed_dir, device, cfg):
    """
    Our purpose is to avoid GPU memory issues by streaming negatives in chunks during mining, rather than trying to mine all negatives for the entire dataset at once.

    Yields:
        (batch_a, batch_p, batch_n, neg_counts, mut_pos_positive, mut_pos_negatives, streaming_info)
    """
    triplets_shuffled = list(triplets)
    random.shuffle(triplets_shuffled)

    for batch_start in range(0, len(triplets_shuffled), cfg.mining_batch_size):
        batch_triplets = triplets_shuffled[batch_start:batch_start + cfg.mining_batch_size]

        anchors = []
        positives = []
        mut_pos_pos_list = []
        valid_triplets = []

        for t in batch_triplets:
            pos_path = random.choice(t['positives'])
            a_data = load_processed_graph_static(t['anchor'], processed_dir)
            p_data = load_processed_graph_static(pos_path, processed_dir)
            if a_data is None or p_data is None:
                continue

            anchors.append(a_data)
            positives.append(p_data)
            mut_pos_pos_list.append(parse_mut_pos_from_path(pos_path))
            valid_triplets.append(t)

        if len(anchors) < 2:
            continue

        batch_a = Batch.from_data_list(anchors).to(device)
        batch_p = Batch.from_data_list(positives).to(device)
        mut_pos_positive = torch.tensor(
            [m if m is not None else -1 for m in mut_pos_pos_list],
            dtype=torch.long
        )

        # Mining pass (no_grad)
        was_training = model.training
        model.eval()

        with torch.no_grad():
            mut_pos_positive_device = mut_pos_positive.to(device)
            ea_mining, _ = model(batch_a, mut_pos=mut_pos_positive_device)
            ep_mining, _ = model(batch_p, mut_pos=mut_pos_positive_device)
            pos_dist_mining = F.pairwise_distance(ea_mining, ep_mining, p=2)

        qual_negs, qual_mut_pos, neg_count_list, streaming_info = mine_negatives_streaming(
            model, ea_mining, pos_dist_mining, valid_triplets,
            processed_dir, device, cfg.margin, cfg.max_negatives,
            cfg.mining_chunk_size
        )

        del ea_mining, ep_mining, pos_dist_mining

        if was_training:
            model.train()

        if not qual_negs:
            continue

        batch_n = Batch.from_data_list(qual_negs)
        neg_counts = torch.tensor(neg_count_list, dtype=torch.long)
        mut_pos_negatives = torch.tensor(qual_mut_pos, dtype=torch.long)

        yield batch_a, batch_p, batch_n, neg_counts, mut_pos_positive, mut_pos_negatives, streaming_info
