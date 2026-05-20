import os
import re

import torch
from torch_geometric.data import Batch


def parse_mut_pos_from_path(path):
    """
    Parse mutation position from CIF filename.
    Example: ..._a109v_... -> 109
    Returns int or None.
    """
    name = os.path.splitext(os.path.basename(path))[0].lower()
    m = re.search(r'_(?:[a-z])(\d+)(?:[a-z])(?:_|$)', name)
    if m and m.group(1) == '0':
        print(f"!!!!Warning: Parsed mutation position 0 from {path} - check filename format")
    return int(m.group(1)) if m else None


def triplet_collate(data_list):
    """
    Collate function for triplet batches with variable number of negatives.
    Each item is (data_a, data_p, data_n_list, mut_pos_positive, mut_pos_negatives).

    Returns:
        batch_a, batch_p, batch_n, neg_counts, mut_pos_positive, mut_pos_negatives
    """
    data_list = [x for x in data_list if x is not None]
    if not data_list:
        return None

    batch_a = Batch.from_data_list([x[0] for x in data_list])
    batch_p = Batch.from_data_list([x[1] for x in data_list])

    all_negatives = []
    neg_counts = []
    mut_pos_negatives_flat = []

    for x in data_list:
        neg_list = x[2]
        neg_counts.append(len(neg_list))
        all_negatives.extend(neg_list)
        for p in x[4]:
            mut_pos_negatives_flat.append(p if p is not None else -1)

    batch_n = Batch.from_data_list(all_negatives)
    neg_counts = torch.tensor(neg_counts, dtype=torch.long)
    mut_pos_positive = torch.tensor([x[3] if x[3] is not None else -1 for x in data_list], dtype=torch.long)
    mut_pos_negatives = torch.tensor(mut_pos_negatives_flat, dtype=torch.long)

    return batch_a, batch_p, batch_n, neg_counts, mut_pos_positive, mut_pos_negatives
