"""Inference helpers for exporting ContVAR graph embeddings.

This module is intentionally independent from the training/export pipeline:
it takes already-built PyG graph ``.pt`` files and a frozen ContVAR checkpoint,
then writes one normalized graph embedding per input graph.
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from contvar.model import DeepProteinGAT


DEFAULT_MODEL_KWARGS = {
    "input_dim": 1300,
    "hidden_dim": 128,
    "output_dim": 256,
    "heads": 4,
    "edge_dim": 20,
    "projection_hidden_dim": None,
}


def _torch_load(path: str, **kwargs):
    """Call torch.load with weights_only=False when supported."""
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _strip_common_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    for prefix in ("module.", "_orig_mod.", "model."):
        if keys and all(key.startswith(prefix) for key in keys):
            return OrderedDict(
                (key[len(prefix):], value) for key, value in state_dict.items()
            )
    return state_dict


def _extract_state_dict(checkpoint) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, torch.nn.Module):
        return None

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a torch.nn.Module, a raw state_dict, or a dict "
            "containing 'model_state_dict'/'state_dict'."
        )

    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_common_prefix(value)

    if all(hasattr(value, "shape") for value in checkpoint.values()):
        return _strip_common_prefix(checkpoint)

    raise ValueError(
        "Could not find model weights in checkpoint. Expected a raw state_dict "
        "or a dict with 'model_state_dict' or 'state_dict'."
    )


def _infer_model_kwargs_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Optional[int]]:
    """Infer ContVAR architecture dimensions from a saved state_dict."""
    inferred: Dict[str, Optional[int]] = {}

    input_proj = state_dict.get("input_proj.weight")
    edge_encoder = state_dict.get("edge_encoder.0.weight")
    projection_0 = state_dict.get("projection.0.weight")
    projection_2 = state_dict.get("projection.2.weight")

    if input_proj is not None:
        inferred["input_dim"] = int(input_proj.shape[1])
        conv_out_dim = int(input_proj.shape[0])
    else:
        conv_out_dim = None

    if edge_encoder is not None:
        inferred["hidden_dim"] = int(edge_encoder.shape[0])
        inferred["edge_dim"] = int(edge_encoder.shape[1])
    else:
        inferred["hidden_dim"] = None

    if projection_0 is not None:
        inferred["projection_hidden_dim"] = int(projection_0.shape[0])

    if projection_2 is not None:
        inferred["output_dim"] = int(projection_2.shape[0])

    hidden_dim = inferred.get("hidden_dim")
    if conv_out_dim is not None and hidden_dim:
        if conv_out_dim % hidden_dim != 0:
            raise ValueError(
                "Cannot infer GAT head count: input_proj output dimension "
                f"{conv_out_dim} is not divisible by hidden_dim {hidden_dim}."
            )
        inferred["heads"] = conv_out_dim // hidden_dim

    return {key: value for key, value in inferred.items() if value is not None}


def load_frozen_contvar_model(
    checkpoint_path: str,
    device: Optional[str] = "auto",
    strict: bool = True,
    model_kwargs: Optional[Dict[str, int]] = None,
) -> DeepProteinGAT:
    """Load a frozen ContVAR model for inference.

    Args:
        checkpoint_path: Path to a saved ``.pt`` file.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or any torch device string.
        strict: Passed to ``load_state_dict`` for state-dict checkpoints.
        model_kwargs: Optional architecture overrides. When omitted, dimensions
            are inferred from the checkpoint where possible and otherwise fall
            back to the repository defaults.

    Returns:
        A ``DeepProteinGAT`` instance in eval mode on ``device``.
    """
    resolved_device = _resolve_device(device)
    checkpoint = _torch_load(checkpoint_path, map_location=resolved_device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(resolved_device)
        model.eval()
        return model

    state_dict = _extract_state_dict(checkpoint)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("mutation_attention_")
    }
    kwargs = dict(DEFAULT_MODEL_KWARGS)
    kwargs.update(_infer_model_kwargs_from_state_dict(state_dict))
    if model_kwargs:
        kwargs.update(
            {key: value for key, value in model_kwargs.items() if value is not None}
        )

    model = DeepProteinGAT(
        input_dim=int(kwargs["input_dim"]),
        hidden_dim=int(kwargs["hidden_dim"]),
        output_dim=int(kwargs["output_dim"]),
        heads=int(kwargs["heads"]),
        edge_dim=int(kwargs["edge_dim"]),
        projection_hidden_dim=kwargs.get("projection_hidden_dim"),
    )
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(
                "[Inference] Missing checkpoint keys ignored: "
                + ", ".join(incompatible.missing_keys[:10])
            )
        if incompatible.unexpected_keys:
            print(
                "[Inference] Unexpected checkpoint keys ignored: "
                + ", ".join(incompatible.unexpected_keys[:10])
            )
    model.to(resolved_device)
    model.eval()
    return model


def embed_batch_global_mean(
    model: DeepProteinGAT,
    batch: Batch,
) -> torch.Tensor:
    """Return normalized graph embeddings using global_mean_pool only."""
    if not hasattr(model, "_gnn_forward") or not hasattr(model, "projection"):
        raise TypeError(
            "The loaded model must expose DeepProteinGAT._gnn_forward and "
            "DeepProteinGAT.projection for global-mean inference."
        )

    x, graph_batch, _ = model._gnn_forward(batch)
    pooled = global_mean_pool(x, graph_batch)
    z = model.projection(pooled)
    return F.normalize(z, p=2, dim=1)


def _iter_batches(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def collect_graph_paths(graph_root: str, recursive: bool = True) -> List[str]:
    """Collect prebuilt PyG ``.pt`` graph paths from a file or directory."""
    if os.path.isfile(graph_root):
        if not graph_root.lower().endswith(".pt"):
            raise ValueError(f"Expected a .pt graph file, got: {graph_root}")
        return [os.path.abspath(graph_root)]

    if not os.path.isdir(graph_root):
        raise FileNotFoundError(f"Graph root not found: {graph_root}")

    graph_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(graph_root):
            for fname in files:
                if fname.lower().endswith(".pt"):
                    graph_paths.append(os.path.abspath(os.path.join(root, fname)))
    else:
        for fname in os.listdir(graph_root):
            path = os.path.join(graph_root, fname)
            if os.path.isfile(path) and fname.lower().endswith(".pt"):
                graph_paths.append(os.path.abspath(path))

    return sorted(graph_paths)


def _read_graph_list(graph_list_path: str) -> List[str]:
    base_dir = os.path.dirname(os.path.abspath(graph_list_path))
    paths: List[str] = []
    with open(graph_list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if not os.path.isabs(value):
                value = os.path.join(base_dir, value)
            paths.append(os.path.abspath(value))
    return paths


def _make_h5_keys(graph_paths: Sequence[str], key_root: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    counts: Dict[str, int] = {}
    root = os.path.abspath(key_root) if key_root else None

    for path in graph_paths:
        if root:
            candidate = os.path.splitext(os.path.relpath(path, root))[0]
        else:
            candidate = os.path.splitext(os.path.basename(path))[0]
        candidate = candidate.replace("\\", "/").replace("/", "__")
        candidate = candidate.strip("_") or "graph"

        count = counts.get(candidate, 0) + 1
        counts[candidate] = count
        keys.append(candidate if count == 1 else f"{candidate}__{count}")

    return keys


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def export_embeddings_from_model(
    model: DeepProteinGAT,
    graph_paths: Sequence[str],
    out_path: str,
    batch_size: int = 32,
    device: Optional[str] = "auto",
    key_root: Optional[str] = None,
) -> str:
    """Export global-mean ContVAR embeddings for prebuilt graph files."""
    if not graph_paths:
        raise ValueError("No graph paths were provided for inference.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()

    graph_paths = [os.path.abspath(path) for path in graph_paths]
    keys = _make_h5_keys(graph_paths, key_root=key_root)
    _ensure_parent_dir(out_path)

    total_batches = (len(graph_paths) + batch_size - 1) // batch_size
    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["embedding_type"] = "contvar_global_mean_pool"
        h5f.attrs["normalized"] = True
        h5f.attrs["num_graphs"] = len(graph_paths)

        for batch_idx, chunk_paths in enumerate(tqdm(
            _iter_batches(graph_paths, batch_size),
            total=total_batches,
            desc=f"Inference -> {os.path.basename(out_path)}",
        )):
            data_list = [
                _torch_load(path, map_location="cpu")
                for path in chunk_paths
            ]
            batch = Batch.from_data_list(data_list).to(resolved_device)
            with torch.inference_mode():
                embeddings = embed_batch_global_mean(model, batch)

            emb_np = embeddings.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            start_idx = batch_idx * batch_size
            for key, source_path, emb in zip(
                keys[start_idx:start_idx + len(chunk_paths)],
                chunk_paths,
                emb_np,
            ):
                dataset = h5f.create_dataset(key, data=emb, compression="gzip")
                dataset.attrs["source_path"] = source_path

    return out_path


def export_batch_embeddings(
    checkpoint_path: str,
    graph_paths: Sequence[str],
    out_path: str,
    batch_size: int = 32,
    device: Optional[str] = "auto",
    strict: bool = True,
    model_kwargs: Optional[Dict[str, int]] = None,
    key_root: Optional[str] = None,
) -> str:
    """Load a frozen checkpoint and export batch graph embeddings."""
    model = load_frozen_contvar_model(
        checkpoint_path=checkpoint_path,
        device=device,
        strict=strict,
        model_kwargs=model_kwargs,
    )
    return export_embeddings_from_model(
        model=model,
        graph_paths=graph_paths,
        out_path=out_path,
        batch_size=batch_size,
        device=device,
        key_root=key_root,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen ContVAR inference on prebuilt PyG .pt graphs and export "
            "global-mean graph embeddings to H5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Frozen model .pt path.")
    parser.add_argument("--out", required=True, help="Output H5 path.")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--graph-root", help="A .pt file or directory of .pt graphs.")
    inputs.add_argument(
        "--graph-list",
        help="Text file containing one .pt graph path per line.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only read .pt files directly under --graph-root.",
    )
    parser.add_argument(
        "--key-root",
        help="Optional root used to build H5 dataset keys from relative paths.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )

    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--output-dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--edge-dim", type=int)
    parser.add_argument("--projection-hidden-dim", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.graph_list:
        graph_paths = _read_graph_list(args.graph_list)
        key_root = args.key_root
    else:
        graph_paths = collect_graph_paths(
            args.graph_root,
            recursive=not args.non_recursive,
        )
        key_root = args.key_root or (
            args.graph_root if os.path.isdir(args.graph_root) else None
        )

    model_kwargs = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "heads": args.heads,
        "edge_dim": args.edge_dim,
        "projection_hidden_dim": args.projection_hidden_dim,
    }

    out_path = export_batch_embeddings(
        checkpoint_path=args.checkpoint,
        graph_paths=graph_paths,
        out_path=args.out,
        batch_size=args.batch_size,
        device=args.device,
        strict=not args.non_strict,
        model_kwargs=model_kwargs,
        key_root=key_root,
    )
    print(
        f"[Inference] Wrote {len(graph_paths)} global-mean embeddings to {out_path}"
    )
    return out_path


if __name__ == "__main__":
    main()
