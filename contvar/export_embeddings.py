"""Internal helpers for exporting ContVAR encoder embeddings after training."""

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from contvar.config import ProjectConfig, ensure_dms_triplets_unzipped
from contvar.data.dataset import TripletProteinGraphDataset
from contvar.data.mapper import TripletDataPathMapper
from contvar.utils import load_all_embeddings


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _batched(
    seq: Sequence[Tuple[str, str]], batch_size: int
) -> Iterable[Sequence[Tuple[str, str]]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _write_embeddings(
    entries: Sequence[Tuple[str, str]],
    model: torch.nn.Module,
    device: torch.device,
    out_path: str,
    batch_size: int,
    loader: Callable[[str], object],
) -> None:
    _ensure_parent_dir(out_path)
    total_batches = (len(entries) + batch_size - 1) // batch_size
    with h5py.File(out_path, "w") as h5f:
        for chunk in tqdm(
            _batched(entries, batch_size),
            total=total_batches,
            desc=f"Exporting -> {os.path.basename(out_path)}",
        ):
            keys = [key for key, _ in chunk]
            data_list = [loader(source) for _, source in chunk]
            batch = Batch.from_data_list(list(data_list)).to(device)
            with torch.no_grad():
                z_global, _ = model(batch)
            emb_np = z_global.detach().cpu().numpy().astype(np.float32, copy=False)
            for key, emb in zip(keys, emb_np):
                h5f.create_dataset(key, data=emb, compression="gzip")


def _load_phase0_targets(split_json_path: str) -> Dict[str, str]:
    with open(split_json_path, "r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    protein_to_split = bundle.get("protein_to_split")
    if not isinstance(protein_to_split, dict) or not protein_to_split:
        raise ValueError(
            f"{split_json_path} must contain a non-empty 'protein_to_split' mapping."
        )
    return protein_to_split


def _index_prebuilt_graphs(prebuilt_graph_root: str) -> Dict[str, str]:
    if not os.path.isdir(prebuilt_graph_root):
        raise FileNotFoundError(
            f"GO prebuilt graph directory not found: {prebuilt_graph_root}"
        )

    index: Dict[str, str] = {}
    duplicates: List[str] = []
    for root, _, files in os.walk(prebuilt_graph_root):
        for fname in files:
            if not fname.lower().endswith(".pt"):
                continue
            base = os.path.splitext(fname)[0]
            prefix = base.split("_", 1)[0].upper()
            if not prefix:
                continue
            path = os.path.join(root, fname)
            if prefix in index and index[prefix] != path:
                duplicates.append(prefix)
                continue
            index[prefix] = path

    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:5])
        print(
            "Warning: duplicate GO graph prefixes detected; keeping the first match for: "
            f"{sample}"
        )
    return index


def export_phase0_embeddings(
    model: torch.nn.Module,
    device: torch.device,
    split_json_path: str,
    prebuilt_graph_root: str,
    out_path: str,
    batch_size: int,
) -> None:
    protein_to_split = _load_phase0_targets(split_json_path)
    graph_index = _index_prebuilt_graphs(prebuilt_graph_root)

    entries: List[Tuple[str, str]] = []
    missing: List[str] = []
    for protein_id in sorted(protein_to_split):
        graph_path = graph_index.get(protein_id.upper())
        if graph_path is None:
            missing.append(protein_id)
            continue
        entries.append((protein_id, graph_path))

    if not entries:
        raise RuntimeError(
            "No GO proteins could be exported. Check the split JSON and prebuilt graph root."
        )

    print(
        f"Phase-0 export: {len(entries)} proteins from {split_json_path} -> {out_path}"
    )
    if missing:
        sample = ", ".join(missing[:5])
        print(
            f"Warning: {len(missing)} GO proteins were skipped because no .pt graph was found. "
            f"Sample: {sample}"
        )

    _write_embeddings(
        entries,
        model,
        device,
        out_path,
        batch_size,
        loader=lambda path: torch.load(path, weights_only=False),
    )


def _collect_dms_targets(
    mapper: TripletDataPathMapper,
    include_anchors: bool,
) -> Dict[str, str]:
    targets: Dict[str, str] = {}
    for triplet in mapper.triplets:
        if include_anchors:
            key = os.path.splitext(os.path.basename(triplet["anchor"]))[0]
            targets[key] = triplet["anchor"]
        for variant_path in triplet["positives"] + triplet["negatives"]:
            key = os.path.splitext(os.path.basename(variant_path))[0]
            targets[key] = variant_path
    return targets


def _processed_graph_path(data_root: str, source_path: str) -> str:
    pdb_code = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(data_root, "processed", f"{pdb_code}.pt")


def _missing_processed_graphs(data_root: str, source_paths: Iterable[str]) -> List[str]:
    missing = []
    for path in source_paths:
        if not os.path.exists(_processed_graph_path(data_root, path)):
            missing.append(path)
    return missing


def export_dms_embeddings(
    model: torch.nn.Module,
    cfg: ProjectConfig,
    device: torch.device,
    data_root: str,
    dms_split_json_path: str,
    out_path: str,
    batch_size: int,
    embeddings_path: Optional[str] = None,
    force_reprocess: bool = False,
    include_anchors: bool = False,
) -> None:
    ensure_dms_triplets_unzipped(data_root, data_zip=None)
    mapper = TripletDataPathMapper(data_root, split_json_path=dms_split_json_path)
    targets = _collect_dms_targets(mapper, include_anchors=include_anchors)
    target_paths = list(targets.values())
    missing_graphs = _missing_processed_graphs(data_root, target_paths)

    if force_reprocess or missing_graphs:
        if not embeddings_path:
            raise ValueError(
                "DMS export needs an embeddings H5 path when processed graphs are missing "
                "or when force_reprocess is used."
            )
        preloaded_embeddings = load_all_embeddings(embeddings_path)
        TripletProteinGraphDataset(
            mapper,
            root=data_root,
            config=cfg,
            split="train",
            esm2_embedding_path=embeddings_path,
            force=force_reprocess,
            preloaded_embeddings=preloaded_embeddings,
        )

    entries: List[Tuple[str, str]] = []
    for key, source_path in sorted(targets.items()):
        graph_path = _processed_graph_path(data_root, source_path)
        if not os.path.exists(graph_path):
            continue
        entries.append((key, graph_path))

    if not entries:
        raise RuntimeError(
            "No DMS graphs were available for export. Check protein_triplets_data/processed."
        )

    print(
        f"DMS export: {len(entries)} embeddings from {dms_split_json_path} -> {out_path}"
    )
    if missing_graphs and not force_reprocess:
        print(
            f"Filled in {len(missing_graphs)} missing processed graphs before export."
        )

    _write_embeddings(
        entries,
        model,
        device,
        out_path,
        batch_size,
        loader=lambda path: torch.load(path, weights_only=False),
    )


def export_all_embeddings(
    model: torch.nn.Module,
    cfg: ProjectConfig,
    device: torch.device,
    data_root: str,
    embeddings_path: Optional[str],
    go_prebuilt_graph_root: Optional[str],
    phase0_split_json_path: str,
    dms_split_json_path: str,
    phase0_out_path: Optional[str],
    dms_out_path: Optional[str],
    batch_size: int = 32,
    force_dms_reprocess: bool = False,
    include_dms_anchors: bool = False,
) -> Dict[str, Optional[str]]:
    results = {
        "phase0_out_path": None,
        "dms_out_path": None,
    }

    print("\n=== Embedding Exports ===")

    if go_prebuilt_graph_root and phase0_out_path:
        export_phase0_embeddings(
            model=model,
            device=device,
            split_json_path=phase0_split_json_path,
            prebuilt_graph_root=go_prebuilt_graph_root,
            out_path=phase0_out_path,
            batch_size=batch_size,
        )
        results["phase0_out_path"] = phase0_out_path
    else:
        print(
            "Skipping phase-0 protein embedding export because go_prebuilt_graph_root is not set."
        )

    if dms_out_path:
        export_dms_embeddings(
            model=model,
            cfg=cfg,
            device=device,
            data_root=data_root,
            dms_split_json_path=dms_split_json_path,
            out_path=dms_out_path,
            batch_size=batch_size,
            embeddings_path=embeddings_path,
            force_reprocess=force_dms_reprocess,
            include_anchors=include_dms_anchors,
        )
        results["dms_out_path"] = dms_out_path

    print("Embedding export completed.")
    return results
