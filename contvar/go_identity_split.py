"""
Phase-0 GO: protein-level train/val/test (merged JSON) and triplet filtering.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple


def _normalize_pid(pid: str) -> str:
    return pid.strip().upper()


def load_protein_to_split_json(path: str) -> Dict[str, str]:
    """
    Load per-protein train/val/test labels from a merged bundle.

    Expected JSON: top-level ``protein_to_split`` mapping
    protein_id -> \"train\" | \"val\" | \"test\".
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Protein split JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("protein_to_split")
    if raw is None:
        raise ValueError(
            f"JSON must contain top-level 'protein_to_split' mapping: {path}"
        )
    out: Dict[str, str] = {}
    for k, v in raw.items():
        sp = str(v).strip().lower()
        if sp not in ("train", "val", "test"):
            continue
        out[_normalize_pid(str(k))] = sp
    return out


def filter_triplets_by_split(
    triplets: List[Tuple[str, str, str]],
    protein_to_split: Dict[str, str],
    split_name: str,
) -> List[Tuple[str, str, str]]:
    """Keep triplets where anchor, positive, negative all map to split_name."""
    out: List[Tuple[str, str, str]] = []
    for a, p, n in triplets:
        sa = protein_to_split.get(_normalize_pid(a))
        sp = protein_to_split.get(_normalize_pid(p))
        sn = protein_to_split.get(_normalize_pid(n))
        if sa == sp == sn == split_name:
            out.append((a, p, n))
    return out
