"""
GOAnnotationDataset for decoder training and evaluation.

Each sample contains:
    emb   [encoder_output_dim]  - z_global from the encoder H5
    label [N_go_terms]          - multi-hot binary vector

Split handling supports two JSON formats:
1. Direct protein-level splits via {"protein_to_split": {...}}
2. Legacy UniRef group splits via {"group_to_split": {...}} plus protein_uniref50.tsv
"""

import csv
import json

import h5py
import torch
from torch.utils.data import Dataset


def _load_split_mapping(uniref_tsv: str, split_json: str) -> dict[str, str]:
    """
    Returns a protein_id -> split ("train" | "val" | "test") mapping.

    Supported split_json formats:
    - {"protein_to_split": {protein_id: split}}
    - {"group_to_split": {group_id: split}} plus uniref_tsv
    """
    with open(split_json, "r", encoding="utf-8") as f:
        split_bundle = json.load(f)

    protein_to_split = split_bundle.get("protein_to_split")
    if isinstance(protein_to_split, dict) and protein_to_split:
        return protein_to_split

    group_to_split = split_bundle.get("group_to_split")
    if not isinstance(group_to_split, dict) or not group_to_split:
        raise ValueError(
            f"{split_json} must contain either 'protein_to_split' or 'group_to_split'."
        )

    protein_to_split = {}
    with open(uniref_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["protein_id"]
            grp = row["group_id"]
            split = group_to_split.get(grp)
            if split is not None:
                protein_to_split[pid] = split

    return protein_to_split


class GOAnnotationDataset(Dataset):

    def __init__(
        self,
        h5_path: str,
        annotations: dict[str, set],
        go_vocab: dict[str, int],
        split: str = "train",
        uniref_tsv: str = "protein_uniref50.tsv",
        split_json: str = "phase0_go_split.json",
    ):
        """
        h5_path     : embeddings.h5  - {uniprot_id: array[encoder_output_dim]}
        annotations : output of parse_goa_tsv()
        go_vocab    : output of build_go_vocab()
        split       : "train" | "val" | "test"
        uniref_tsv  : protein_id -> UniRef50 cluster mapping file
        split_json  : protein or UniRef split assignments
        """
        n_classes = len(go_vocab)
        protein_to_split = _load_split_mapping(uniref_tsv, split_json)

        with h5py.File(h5_path, "r") as h5:
            common = sorted(set(h5.keys()) & set(annotations.keys()))
            protein_ids = [
                pid for pid in common
                if protein_to_split.get(pid) == split
            ]

            print(f"[{split}] Loading {len(protein_ids)} proteins...", flush=True)
            embs = torch.stack([
                torch.from_numpy(h5[pid][:]).float()
                for pid in protein_ids
            ]) if protein_ids else torch.empty((0, 0), dtype=torch.float32)

        labels = torch.zeros(len(protein_ids), n_classes, dtype=torch.float32)
        for i, pid in enumerate(protein_ids):
            for go in annotations.get(pid, set()):
                if go in go_vocab:
                    labels[i, go_vocab[go]] = 1.0

        self.embs = embs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embs[idx], self.labels[idx]
