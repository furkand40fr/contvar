"""
GOAnnotationDataset

Each sample contains:
    emb   [encoder_output_dim]  -- protein embedding
    label [N_go_terms]          -- multi-hot binary vector

label[i] = 1  ->  protein has this GO term
label[i] = 0  ->  protein does not have this GO term

Split: UniRef50 cluster-based -- proteins in the same cluster fall into the same split.
Split assignments are read from phase0_go_split.json (train 80% / val 10% / test 10%).
Protein -> cluster mapping is loaded from protein_uniref50.tsv.

Supports three embedding modes:
    "esm"     -> ESM-2 only (1280d)
    "contvar" -> ContVAR GNN only (256d)
    "concat"  -> ESM + ContVAR concatenated (1536d)

All embeddings and labels are loaded into RAM in __init__;
__getitem__ operates with pure tensor indexing (no disk I/O).
"""

import csv
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_uniref_split(uniref_tsv: str, split_json: str) -> dict[str, str]:
    """
    Returns a protein_id -> split ("train" | "val" | "test") mapping.

    uniref_tsv  : file with protein_id <TAB> group_id rows
    split_json  : phase0_go_split.json  (group_id -> split assignments)
    """
    with open(split_json) as f:
        phase0 = json.load(f)
    group_to_split: dict[str, str] = phase0["group_to_split"]

    protein_to_split: dict[str, str] = {}
    with open(uniref_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid   = row["protein_id"]
            grp   = row["group_id"]
            split = group_to_split.get(grp)
            if split is not None:
                protein_to_split[pid] = split

    return protein_to_split


class GOAnnotationDataset(Dataset):

    def __init__(self, esm_h5_path: str, contvar_h5_path: str, annotations: dict[str, set],
                 go_vocab: dict[str, int], split: str = "train",
                 uniref_tsv: str = "protein_uniref50.tsv",
                 split_json: str = "phase0_go_split.json",
                 embedding_type: str = "concat",
                 contvar_full_h5_path: str = None):
        """
        esm_h5_path     : ESM-2 embeddings.h5
        contvar_h5_path : ContVAR GNN embeddings.h5
        annotations : output of parse_goa_tsv()
        go_vocab    : output of build_go_vocab()
        split       : "train" | "val" | "test"
        uniref_tsv  : protein_id -> UniRef50 cluster mapping file
        split_json  : UniRef50 cluster -> split assignments (phase0_go_split.json)
        embedding_type : "esm" | "contvar" | "contvar_full" | "concat" | "concat_full"
        contvar_full_h5_path : ContVAR GNN embeddings.h5 for 'contvar_full'
        """
        n_classes = len(go_vocab)

        protein_to_split = _load_uniref_split(uniref_tsv, split_json)

        # Determine which H5 files to open and how to build embeddings
        if embedding_type == "esm":
            with h5py.File(esm_h5_path, "r") as h5_esm:
                available = sorted(set(h5_esm.keys()) & set(annotations.keys()))
                protein_ids = [
                    pid for pid in available
                    if protein_to_split.get(pid) == split
                ]
                print(f"[{split}] Loading {len(protein_ids)} proteins (ESM-only)...", flush=True)
                embs = torch.stack([
                    torch.from_numpy(h5_esm[pid][:]).float()
                    for pid in protein_ids
                ])

        elif embedding_type == "contvar":
            with h5py.File(contvar_h5_path, "r") as h5_cv:
                available = sorted(set(h5_cv.keys()) & set(annotations.keys()))
                protein_ids = [
                    pid for pid in available
                    if protein_to_split.get(pid) == split
                ]
                print(f"[{split}] Loading {len(protein_ids)} proteins (ContVAR-only)...", flush=True)
                embs = torch.stack([
                    torch.from_numpy(h5_cv[pid][:]).float()
                    for pid in protein_ids
                ])

        elif embedding_type == "contvar_full":
            with h5py.File(contvar_full_h5_path, "r") as h5_cv:
                available = sorted(set(h5_cv.keys()) & set(annotations.keys()))
                protein_ids = [
                    pid for pid in available
                    if protein_to_split.get(pid) == split
                ]
                print(f"[{split}] Loading {len(protein_ids)} proteins (ContVAR-Full-only)...", flush=True)
                embs = torch.stack([
                    torch.from_numpy(h5_cv[pid][:]).float()
                    for pid in protein_ids
                ])

        elif embedding_type == "concat":
            with h5py.File(esm_h5_path, "r") as h5_esm, h5py.File(contvar_h5_path, "r") as h5_cv:
                esm_keys = set(h5_esm.keys())
                cv_keys = set(h5_cv.keys())
                common = sorted(esm_keys & cv_keys & set(annotations.keys()))
                protein_ids = [
                    pid for pid in common
                    if protein_to_split.get(pid) == split
                ]
                print(f"[{split}] Loading {len(protein_ids)} proteins (ESM+ContVAR)...", flush=True)
                embs = torch.stack([
                    torch.from_numpy(np.concatenate([h5_esm[pid][:], h5_cv[pid][:]])).float()
                    for pid in protein_ids
                ])

        elif embedding_type == "concat_full":
            with h5py.File(esm_h5_path, "r") as h5_esm, h5py.File(contvar_full_h5_path, "r") as h5_cv:
                esm_keys = set(h5_esm.keys())
                cv_keys = set(h5_cv.keys())
                common = sorted(esm_keys & cv_keys & set(annotations.keys()))
                protein_ids = [
                    pid for pid in common
                    if protein_to_split.get(pid) == split
                ]
                print(f"[{split}] Loading {len(protein_ids)} proteins (ESM+ContVAR-Full)...", flush=True)
                embs = torch.stack([
                    torch.from_numpy(np.concatenate([h5_esm[pid][:], h5_cv[pid][:]])).float()
                    for pid in protein_ids
                ])

        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        labels = torch.zeros(len(protein_ids), n_classes)
        for i, pid in enumerate(protein_ids):
            for go in annotations.get(pid, set()):
                if go in go_vocab:
                    labels[i, go_vocab[go]] = 1.0

        self.embs   = embs    # [N, D]  -- float32
        self.labels = labels  # [N, C]  -- float32 multi-hot

    def __len__(self) -> int:
        return len(self.embs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embs[idx], self.labels[idx]
