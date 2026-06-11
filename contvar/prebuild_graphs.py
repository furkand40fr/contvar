"""Local graph prebuilding and frozen-checkpoint inference orchestration."""

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBIO
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from contvar.config import ProjectConfig


def _as_path_list(values: Sequence[os.PathLike | str]) -> List[Path]:
    return [Path(value).expanduser().resolve() for value in values if value]


def _find_cif_files(structure_dirs: Sequence[os.PathLike | str]) -> List[Path]:
    cif_paths: List[Path] = []
    for structure_dir in _as_path_list(structure_dirs):
        if structure_dir.is_file() and structure_dir.suffix.lower() == ".cif":
            cif_paths.append(structure_dir)
        elif structure_dir.is_dir():
            cif_paths.extend(sorted(structure_dir.rglob("*.cif")))
    return sorted(dict.fromkeys(cif_paths))


def _protein_key_from_stem(stem: str) -> str:
    key = stem.lower()
    if key.endswith("_model"):
        key = key[:-6]
    return key


def _extract_embedding_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        preferred_keys = (
            "embedding",
            "embeddings",
            "esm2",
            "representations",
            "residue_embeddings",
        )
        for key in preferred_keys:
            if key in value:
                extracted = _extract_embedding_array(value[key])
                if extracted is not None:
                    return extracted
        for candidate in value.values():
            extracted = _extract_embedding_array(candidate)
            if extracted is not None and extracted.ndim == 2:
                return extracted
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return arr
    return None


class LocalEmbeddingStore:
    """Lazy local lookup for per-residue ESM2 embeddings from H5, .pt files, or ZIPs."""

    def __init__(
        self,
        embeddings_h5: Optional[os.PathLike | str] = None,
        embeddings_dir: Optional[os.PathLike | str] = None,
        embeddings_zip: Optional[os.PathLike | str] = None,
    ):
        h5_path = Path(embeddings_h5).expanduser().resolve() if embeddings_h5 else None
        dir_path = Path(embeddings_dir).expanduser().resolve() if embeddings_dir else None
        zip_path = Path(embeddings_zip).expanduser().resolve() if embeddings_zip else None
        if h5_path and h5_path.suffix.lower() == ".zip" and zip_path is None:
            zip_path = h5_path
            h5_path = None
        if dir_path and dir_path.suffix.lower() == ".zip" and zip_path is None:
            zip_path = dir_path
            dir_path = None

        self.embeddings_h5 = h5_path
        self.embeddings_dir = dir_path
        self.embeddings_zip = zip_path
        self._h5 = None
        self._pt_index: Optional[Dict[str, Path]] = None
        self._zip_extract_dir: Optional[Path] = None
        self._zip_prepared = False

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _prepare_zip(self) -> None:
        if self._zip_prepared:
            return
        self._zip_prepared = True
        if not self.embeddings_zip:
            return
        if not self.embeddings_zip.is_file():
            raise FileNotFoundError(f"ESM2 embeddings ZIP not found: {self.embeddings_zip}")

        extract_root = (
            self.embeddings_zip.parent / f"{self.embeddings_zip.stem}_unzipped"
        ).resolve()
        self._zip_extract_dir = extract_root
        with zipfile.ZipFile(self.embeddings_zip, "r") as zf:
            for member in zf.infolist():
                target = (extract_root / member.filename).resolve()
                if extract_root != target and extract_root not in target.parents:
                    raise ValueError(
                        f"Unsafe path inside embeddings ZIP: {member.filename}"
                    )
            if not extract_root.is_dir() or not any(extract_root.iterdir()):
                extract_root.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_root)

        h5_files = sorted(
            path
            for path in extract_root.rglob("*")
            if path.suffix.lower() in {".h5", ".hdf5"} and path.is_file()
        )
        pt_files = sorted(extract_root.rglob("*.pt"))
        if self.embeddings_h5 is None and h5_files:
            self.embeddings_h5 = h5_files[0]
        if self.embeddings_dir is None and pt_files:
            self.embeddings_dir = extract_root

    def _open_h5(self):
        self._prepare_zip()
        if self.embeddings_h5 and self.embeddings_h5.is_file() and self._h5 is None:
            self._h5 = h5py.File(self.embeddings_h5, "r")
        return self._h5

    def _build_pt_index(self) -> Dict[str, Path]:
        if self._pt_index is not None:
            return self._pt_index

        self._prepare_zip()
        index: Dict[str, Path] = {}
        if self.embeddings_dir and self.embeddings_dir.is_dir():
            for path in self.embeddings_dir.rglob("*.pt"):
                index[_protein_key_from_stem(path.stem)] = path
        self._pt_index = index
        return index

    def get(self, protein_stem: str):
        key = _protein_key_from_stem(protein_stem)

        h5_file = self._open_h5()
        if h5_file is not None:
            for candidate in (
                key,
                key.upper(),
                protein_stem,
                protein_stem.lower(),
                protein_stem.upper(),
            ):
                if candidate in h5_file:
                    return np.asarray(h5_file[candidate])

        pt_path = self._build_pt_index().get(key)
        if pt_path is not None:
            value = torch.load(pt_path, map_location="cpu", weights_only=False)
            return _extract_embedding_array(value)

        return None


class LocalGraphPrebuilder:
    def __init__(
        self,
        output_dir: os.PathLike | str,
        config: Optional[ProjectConfig] = None,
        embeddings_h5: Optional[os.PathLike | str] = None,
        embeddings_dir: Optional[os.PathLike | str] = None,
        embeddings_zip: Optional[os.PathLike | str] = None,
        require_embeddings: bool = True,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ProjectConfig()
        self.require_embeddings = require_embeddings
        self.embedding_store = LocalEmbeddingStore(
            embeddings_h5=embeddings_h5,
            embeddings_dir=embeddings_dir,
            embeddings_zip=embeddings_zip,
        )
        self.node_metadata_funcs = self.config.get_active_node_metadata_funcs()
        self.node_attributes = self.config.get_node_attributes_list()
        self.salad_edge_builder = (
            self.config.get_salad_edge_builder()
            if self.config.edge_mode == "salad"
            else None
        )
        self.edge_funcs = (
            []
            if self.config.edge_mode == "salad"
            else self.config.get_active_edge_funcs()
        )

    def close(self) -> None:
        self.embedding_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def build_one(self, cif_path: os.PathLike | str) -> Data:
        cif_path = Path(cif_path).expanduser().resolve()
        protein_code = cif_path.stem
        temp_pdb_path = None

        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_code, str(cif_path))

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                temp_pdb_path = tmp.name

            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(temp_pdb_path)

            graph_config = ProteinGraphConfig(
                edge_construction_functions=self.edge_funcs,
                node_metadata_functions=self.node_metadata_funcs,
                verbose=False,
            )
            graph = construct_graph(
                config=graph_config,
                path=temp_pdb_path,
                verbose=False,
            )
            if graph is None or len(graph.nodes()) == 0:
                raise ValueError("constructed graph is empty")

            first_node = list(graph.nodes())[0]
            chain_id = first_node.split(":")[0]
            protein_embedding = self.embedding_store.get(protein_code)
            if self.config.use_embedding and protein_embedding is None and self.require_embeddings:
                raise KeyError(f"missing ESM2 embedding for {protein_code}")

            self._attach_node_attributes(
                graph=graph,
                chain_id=chain_id,
                protein_code=protein_code,
                protein_embedding=protein_embedding,
            )
            data = self._create_pyg_data(graph)
            data.protein_id = protein_code
            data.source_path = str(cif_path)
            return data
        finally:
            if temp_pdb_path and os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)

    def _attach_node_attributes(self, graph, chain_id, protein_code, protein_embedding):
        if protein_embedding is not None:
            protein_embedding = np.asarray(protein_embedding)
            node_order = []
            for node_name, _ in graph.nodes(data=True):
                parts = node_name.split(":")
                node_order.append((parts[0], int(parts[2]), node_name))
            node_order.sort(key=lambda item: (item[0], item[1]))
            if len(node_order) != len(protein_embedding):
                raise ValueError(
                    f"embedding length mismatch for {protein_code}: "
                    f"{len(protein_embedding)} embeddings vs {len(node_order)} residues"
                )
            name_to_emb_idx = {
                name: idx for idx, (_, _, name) in enumerate(node_order)
            }
        else:
            name_to_emb_idx = {}

        for node_name, attrs in graph.nodes(data=True):
            parts = node_name.split(":")
            attrs["chain_id"] = chain_id
            attrs["residue_name"] = parts[1]
            attrs["residue_number"] = int(parts[2])
            if protein_embedding is not None:
                attrs["embedding"] = protein_embedding[name_to_emb_idx[node_name]]

        if self.config.edge_mode == "graphein":
            for source, target, attrs in graph.edges(data=True):
                source_coords = graph.nodes[source]["coords"]
                target_coords = graph.nodes[target]["coords"]
                attrs["euclidean_distance"] = round(
                    np.sqrt(np.sum(np.square(source_coords - target_coords))).item(),
                    5,
                )

    def _create_pyg_data(self, graph, to_undirected_graph: bool = True) -> Data:
        if self.config.edge_mode == "salad":
            return self._create_pyg_data_salad(graph, to_undirected_graph)
        return self._create_pyg_data_graphein(graph, to_undirected_graph)

    def _create_pyg_data_salad(self, graph, to_undirected_graph: bool) -> Data:
        node_features = []
        coords_list = []
        residue_indices = []
        chain_ids = []

        for index, (_, attrs) in enumerate(graph.nodes(data=True)):
            features = []
            for attr_name in self.node_attributes:
                value = attrs.get(attr_name)
                if value is None:
                    continue
                if isinstance(value, (list, tuple, np.ndarray)):
                    features.extend(list(value))
                else:
                    features.append(value)
            node_features.append(features)
            coords_list.append(attrs["coords"])
            residue_indices.append(attrs.get("residue_number", index))
            chain_ids.append(attrs.get("chain_id", "A"))

        data = Data()
        data.x = torch.tensor(node_features, dtype=torch.float)
        data.pos = torch.tensor(np.asarray(coords_list), dtype=torch.float)

        unique_chains = sorted(set(chain_ids))
        chain_int = np.array([unique_chains.index(chain) for chain in chain_ids])
        edge_index, edge_attr = self.salad_edge_builder.build_edge_index_and_attr(
            coords=np.asarray(coords_list),
            residue_indices=np.asarray(residue_indices),
            chain_ids=chain_int,
            use_rbf=True,
            num_rbf=self.config.salad_num_rbf,
            d_max=self.config.salad_d_max,
        )
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.residue_number = torch.tensor(residue_indices, dtype=torch.long)

        if to_undirected_graph and data.edge_index.numel() > 0:
            data.edge_index, data.edge_attr = to_undirected(
                data.edge_index,
                data.edge_attr,
            )
        return data

    def _create_pyg_data_graphein(self, graph, to_undirected_graph: bool) -> Data:
        node_indexes_mapping = {}
        node_features = []
        residue_numbers = []

        for index, (node_name, attrs) in enumerate(graph.nodes(data=True)):
            features = []
            for attr_name in self.node_attributes:
                value = attrs.get(attr_name)
                if value is None:
                    continue
                if isinstance(value, (list, tuple, np.ndarray)):
                    features.extend(list(value))
                else:
                    features.append(value)
            node_features.append(features)
            node_indexes_mapping[node_name] = index
            residue_numbers.append(attrs.get("residue_number", index))

        edge_index_values = []
        edge_attr_values = []
        for source, target, attrs in graph.edges(data=True):
            edge_index_values.append(
                [node_indexes_mapping[source], node_indexes_mapping[target]]
            )
            edge_attr_values.append([attrs["euclidean_distance"]])

        data = Data()
        data.x = torch.tensor(node_features, dtype=torch.float)
        data.residue_number = torch.tensor(residue_numbers, dtype=torch.long)
        if edge_index_values:
            data.edge_index = torch.tensor(
                edge_index_values,
                dtype=torch.long,
            ).t().contiguous()
            data.edge_attr = torch.tensor(edge_attr_values, dtype=torch.float)
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, self.config.edge_attr_dim), dtype=torch.float)

        if to_undirected_graph and data.edge_index.numel() > 0:
            data.edge_index, data.edge_attr = to_undirected(
                data.edge_index,
                data.edge_attr,
            )
        return data


def _select_chunk(paths: Sequence[Path], total_chunks: int, chunk_id: int) -> List[Path]:
    if total_chunks <= 1:
        return list(paths)
    if chunk_id < 0 or chunk_id >= total_chunks:
        raise ValueError("chunk_id must be in [0, total_chunks)")
    return [path for idx, path in enumerate(paths) if idx % total_chunks == chunk_id]


def _unique_h5_key(path: Path, counts: Dict[str, int]) -> str:
    key = path.stem.replace("\\", "/").replace("/", "__").strip("_") or "graph"
    count = counts.get(key, 0) + 1
    counts[key] = count
    return key if count == 1 else f"{key}__{count}"


def _write_streamed_embeddings(
    models: Dict[str, torch.nn.Module],
    h5_files: Dict[str, h5py.File],
    graph_items: Sequence[Tuple[str, Path, Data]],
    device: torch.device,
) -> None:
    if not graph_items:
        return

    from contvar.inference import embed_batch_global_mean

    keys = [key for key, _, _ in graph_items]
    graph_paths = [str(path) for _, path, _ in graph_items]
    data_list = [data for _, _, data in graph_items]
    batch = Batch.from_data_list(data_list).to(device)

    with torch.inference_mode():
        for checkpoint_path, model in models.items():
            embeddings = embed_batch_global_mean(model, batch)
            emb_np = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
            h5_file = h5_files[checkpoint_path]
            for key, graph_path, emb in zip(keys, graph_paths, emb_np):
                dataset = h5_file.create_dataset(key, data=emb, compression="gzip")
                dataset.attrs["source_path"] = graph_path


def run_prebuild(
    structure_dirs: Sequence[os.PathLike | str],
    output_dir: os.PathLike | str,
    embeddings_h5: Optional[os.PathLike | str] = None,
    embeddings_dir: Optional[os.PathLike | str] = None,
    embeddings_zip: Optional[os.PathLike | str] = None,
    edge_mode: str = "salad",
    force: bool = False,
    total_chunks: int = 1,
    chunk_id: int = 0,
    require_embeddings: bool = True,
) -> Dict[str, object]:
    """Build local PyG graph .pt files from local CIF folders."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    all_cifs = _find_cif_files(structure_dirs)
    selected_cifs = _select_chunk(all_cifs, total_chunks=total_chunks, chunk_id=chunk_id)

    cfg = ProjectConfig()
    cfg.edge_mode = edge_mode

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }
    failures = []
    output_graphs = []

    with LocalGraphPrebuilder(
        output_dir=output_path,
        config=cfg,
        embeddings_h5=embeddings_h5,
        embeddings_dir=embeddings_dir,
        embeddings_zip=embeddings_zip,
        require_embeddings=require_embeddings,
    ) as builder:
        for cif_path in tqdm(selected_cifs, desc="Prebuilding graphs"):
            out_path = output_path / f"{cif_path.stem}.pt"
            if out_path.exists() and not force:
                stats["skipped"] += 1
                output_graphs.append(str(out_path))
                continue
            try:
                data = builder.build_one(cif_path)
                torch.save(data, out_path)
                stats["processed"] += 1
                output_graphs.append(str(out_path))
            except Exception as exc:
                stats["failed"] += 1
                failures.append(
                    {
                        "source_path": str(cif_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    metadata = {
        "structure_dirs": [str(path) for path in _as_path_list(structure_dirs)],
        "output_dir": str(output_path),
        "embeddings_h5": str(Path(embeddings_h5).resolve()) if embeddings_h5 else None,
        "embeddings_dir": str(Path(embeddings_dir).resolve()) if embeddings_dir else None,
        "embeddings_zip": str(Path(embeddings_zip).resolve()) if embeddings_zip else None,
        "edge_mode": edge_mode,
        "total_cifs": len(all_cifs),
        "selected_cifs": len(selected_cifs),
        "total_chunks": total_chunks,
        "chunk_id": chunk_id,
        "unique_proteins": len({path.stem for path in all_cifs}),
        "stats": stats,
        "failures": failures,
        "output_graphs": output_graphs,
    }
    meta_path = output_path / "prebuild_metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Prebuild complete: processed={stats['processed']}, "
        f"skipped={stats['skipped']}, failed={stats['failed']}"
    )
    print(f"Metadata: {meta_path}")
    return metadata


def run_prebuild_and_inference(
    structure_dirs: Sequence[os.PathLike | str],
    output_dir: os.PathLike | str,
    checkpoints: Sequence[os.PathLike | str],
    inference_output_dir: os.PathLike | str,
    embeddings_h5: Optional[os.PathLike | str] = None,
    embeddings_dir: Optional[os.PathLike | str] = None,
    embeddings_zip: Optional[os.PathLike | str] = None,
    edge_mode: str = "salad",
    force: bool = False,
    total_chunks: int = 1,
    chunk_id: int = 0,
    require_embeddings: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    strict: bool = True,
    save_graphs: bool = True,
) -> Dict[str, object]:
    """Build/cache graphs locally and stream each batch through checkpoint inference."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    inference_dir = Path(inference_output_dir).expanduser().resolve()
    inference_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [Path(path).expanduser().resolve() for path in checkpoints]
    missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(missing_checkpoints)
        )

    from contvar.inference import _resolve_device, load_frozen_contvar_model

    resolved_device = _resolve_device(device)
    models: Dict[str, torch.nn.Module] = {}
    inference_outputs: Dict[str, str] = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint_key = str(checkpoint_path)
        models[checkpoint_key] = load_frozen_contvar_model(
            checkpoint_path=checkpoint_key,
            device=device,
            strict=strict,
        )
        inference_outputs[checkpoint_key] = str(
            inference_dir / f"{checkpoint_path.stem}_contvar_embeddings.h5"
        )

    all_cifs = _find_cif_files(structure_dirs)
    selected_cifs = _select_chunk(all_cifs, total_chunks=total_chunks, chunk_id=chunk_id)

    cfg = ProjectConfig()
    cfg.edge_mode = edge_mode
    stats = {
        "processed": 0,
        "cached": 0,
        "failed": 0,
        "embedded": 0,
        "saved": 0,
    }
    failures = []
    output_graphs = []
    pending: List[Tuple[str, Path, Data]] = []
    key_counts: Dict[str, int] = {}
    meta_path = output_path / "prebuild_metadata.json"

    def build_metadata(status: str) -> Dict[str, object]:
        return {
            "status": status,
            "structure_dirs": [str(path) for path in _as_path_list(structure_dirs)],
            "output_dir": str(output_path),
            "inference_output_dir": str(inference_dir),
            "embeddings_h5": str(Path(embeddings_h5).resolve()) if embeddings_h5 else None,
            "embeddings_dir": str(Path(embeddings_dir).resolve()) if embeddings_dir else None,
            "embeddings_zip": str(Path(embeddings_zip).resolve()) if embeddings_zip else None,
            "edge_mode": edge_mode,
            "save_graphs": save_graphs,
            "total_cifs": len(all_cifs),
            "selected_cifs": len(selected_cifs),
            "total_chunks": total_chunks,
            "chunk_id": chunk_id,
            "unique_proteins": len({path.stem for path in all_cifs}),
            "stats": dict(stats),
            "failures": list(failures),
            "output_graphs": list(output_graphs),
            "inference_outputs": inference_outputs,
        }

    def write_metadata(path: Path, status: str) -> Dict[str, object]:
        metadata = build_metadata(status=status)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return metadata

    print(f"Metadata path: {meta_path}")

    h5_files: Dict[str, h5py.File] = {}
    try:
        for checkpoint_key, out_path in inference_outputs.items():
            h5_file = h5py.File(out_path, "w")
            h5_file.attrs["embedding_type"] = "contvar_global_mean_pool"
            h5_file.attrs["normalized"] = True
            h5_file.attrs["checkpoint_path"] = checkpoint_key
            h5_file.attrs["streaming_graph_cache"] = save_graphs
            h5_files[checkpoint_key] = h5_file

        with LocalGraphPrebuilder(
            output_dir=output_path,
            config=cfg,
            embeddings_h5=embeddings_h5,
            embeddings_dir=embeddings_dir,
            embeddings_zip=embeddings_zip,
            require_embeddings=require_embeddings,
        ) as builder:
            progress = tqdm(selected_cifs, desc="Prebuild + inference")
            for index, cif_path in enumerate(progress, start=1):
                graph_path = output_path / f"{cif_path.stem}.pt"
                try:
                    if graph_path.exists() and not force:
                        data = torch.load(
                            graph_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        stats["cached"] += 1
                    else:
                        data = builder.build_one(cif_path)
                        if save_graphs:
                            torch.save(data, graph_path)
                            stats["saved"] += 1
                        stats["processed"] += 1
                except Exception as exc:
                    stats["failed"] += 1
                    failures.append(
                        {
                            "source_path": str(cif_path),
                            "graph_path": str(graph_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if index == 1 or index % 25 == 0 or index == len(selected_cifs):
                        progress.set_postfix(
                            processed=stats["processed"],
                            cached=stats["cached"],
                            saved=stats["saved"],
                            embedded=stats["embedded"],
                            failed=stats["failed"],
                            pending=len(pending),
                            refresh=False,
                        )

                    continue

                key = _unique_h5_key(graph_path, key_counts)
                source_path = graph_path if graph_path.exists() else cif_path
                pending.append((key, source_path, data))
                if graph_path.exists():
                    output_graphs.append(str(graph_path))

                if len(pending) >= batch_size:
                    _write_streamed_embeddings(
                        models=models,
                        h5_files=h5_files,
                        graph_items=pending,
                        device=resolved_device,
                    )
                    stats["embedded"] += len(pending)
                    pending.clear()

                if index == 1 or index % 25 == 0 or index == len(selected_cifs):
                    progress.set_postfix(
                        processed=stats["processed"],
                        cached=stats["cached"],
                        saved=stats["saved"],
                        embedded=stats["embedded"],
                        failed=stats["failed"],
                        pending=len(pending),
                        refresh=False,
                    )

            if pending:
                _write_streamed_embeddings(
                    models=models,
                    h5_files=h5_files,
                    graph_items=pending,
                    device=resolved_device,
                )
                stats["embedded"] += len(pending)
                pending.clear()
                progress.set_postfix(
                    processed=stats["processed"],
                    cached=stats["cached"],
                    saved=stats["saved"],
                    embedded=stats["embedded"],
                    failed=stats["failed"],
                    pending=0,
                    refresh=False,
                )

        for h5_file in h5_files.values():
            h5_file.attrs["num_graphs"] = stats["embedded"]
    finally:
        for h5_file in h5_files.values():
            h5_file.close()

    metadata = write_metadata(
        meta_path,
        status="failed" if stats["embedded"] == 0 else "complete",
    )

    print(
        f"Streaming prebuild+inference complete: processed={stats['processed']}, "
        f"cached={stats['cached']}, saved={stats['saved']}, "
        f"embedded={stats['embedded']}, failed={stats['failed']}"
    )
    print(f"Metadata: {meta_path}")
    if stats["embedded"] == 0:
        sample_errors = "; ".join(
            failure["error"] for failure in failures[:5]
        ) or "no per-file failures were recorded"
        raise RuntimeError(
            f"No graph embeddings were produced. Metadata was written to {meta_path}. "
            f"First failure(s): {sample_errors}"
        )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/cache local ContVAR graphs and optionally stream them through "
            "checkpoint embedding export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--structure-dir",
        action="append",
        required=True,
        help="Local CIF file or directory. Repeat for multiple roots.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for graph .pt files.")
    parser.add_argument("--embeddings-h5", help="Optional local per-protein ESM2 H5 file.")
    parser.add_argument("--embeddings-dir", help="Optional local directory of per-protein .pt embeddings.")
    parser.add_argument(
        "--embeddings-zip",
        help="Optional ZIP containing an ESM2 H5 file or per-protein .pt embeddings.",
    )
    parser.add_argument("--edge-mode", default="salad", choices=("salad", "graphein"))
    parser.add_argument("--force", action="store_true", help="Regenerate existing graph .pt files.")
    parser.add_argument(
        "--no-save-graphs",
        action="store_true",
        help="Build graphs in memory for inference without writing new .pt graph cache files.",
    )
    parser.add_argument("--total-chunks", type=int, default=1)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument(
        "--allow-missing-embeddings",
        action="store_true",
        help="Allow 20-dimensional amino-acid-only graphs when ESM2 embeddings are missing.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        help=(
            "Checkpoint to stream graphs through. Repeat for two or more checkpoints."
        ),
    )
    parser.add_argument(
        "--inference-output-dir",
        default=None,
        help="Directory for H5 embedding exports. Defaults to output-dir/embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.checkpoint:
        return run_prebuild_and_inference(
            structure_dirs=args.structure_dir,
            output_dir=args.output_dir,
            checkpoints=args.checkpoint,
            inference_output_dir=args.inference_output_dir
            or str(Path(args.output_dir) / "embeddings"),
            embeddings_h5=args.embeddings_h5,
            embeddings_dir=args.embeddings_dir,
            embeddings_zip=args.embeddings_zip,
            edge_mode=args.edge_mode,
            force=args.force,
            total_chunks=args.total_chunks,
            chunk_id=args.chunk_id,
            require_embeddings=not args.allow_missing_embeddings,
            batch_size=args.batch_size,
            device=args.device,
            strict=not args.non_strict,
            save_graphs=not args.no_save_graphs,
        )

    return run_prebuild(
        structure_dirs=args.structure_dir,
        output_dir=args.output_dir,
        embeddings_h5=args.embeddings_h5,
        embeddings_dir=args.embeddings_dir,
        embeddings_zip=args.embeddings_zip,
        edge_mode=args.edge_mode,
        force=args.force,
        total_chunks=args.total_chunks,
        chunk_id=args.chunk_id,
        require_embeddings=not args.allow_missing_embeddings,
    )


if __name__ == "__main__":
    main()
