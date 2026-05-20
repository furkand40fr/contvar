import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch_geometric.data import Batch

from contvar.config import ProjectConfig
from contvar.data.go_dataset import GOSemanticTripletDataset
from contvar.go_identity_split import load_protein_to_split_json
from contvar.metrics import compute_detailed_metrics

# Ontology order matches meeting pseudocode (MF → BP → CC).
_GO_ONTOLOGY_ORDER: Tuple[str, ...] = ("mf", "bp", "cc")
_GO_DETAIL_METRIC_NAMES: Tuple[str, ...] = ("Alignment", "Uniformity", "MRR")
_GO_SUMMARIZED_METRIC_NAMES: Tuple[str, ...] = ("loss",) + _GO_DETAIL_METRIC_NAMES


def _triplet_loss(anchor, positive, negative, margin: float):
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_pos - d_neg + margin).mean(), d_pos, d_neg


def _save_model_checkpoint(model, checkpoint_path: Optional[str]):
    """Persist the current model weights if a checkpoint path is configured."""
    if not checkpoint_path:
        return
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def _infinite_batches(loader: DataLoader):
    """Yield batches forever; each pass over the loader gets a fresh shuffle."""
    while True:
        for batch in loader:
            yield batch


def _normalize_sampling_ratio(
    raw_ratio, active_ontologies: List[str]
) -> Dict[str, float]:
    """
    Build a valid probability map over active ontologies.
    Supports dict-like {"mf": 0.6, "bp": 0.2, "cc": 0.2} or list/tuple in
    ontology order (mf, bp, cc). Falls back to uniform if invalid.
    """
    if not active_ontologies:
        return {}

    weights: Dict[str, float] = {ont: 0.0 for ont in active_ontologies}

    if isinstance(raw_ratio, dict):
        for ont in active_ontologies:
            try:
                weights[ont] = max(float(raw_ratio.get(ont, 0.0)), 0.0)
            except (TypeError, ValueError):
                weights[ont] = 0.0
    elif isinstance(raw_ratio, (tuple, list)):
        for ont, w in zip(_GO_ONTOLOGY_ORDER, raw_ratio):
            if ont not in weights:
                continue
            try:
                weights[ont] = max(float(w), 0.0)
            except (TypeError, ValueError):
                weights[ont] = 0.0

    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / float(len(active_ontologies))
        return {ont: uniform for ont in active_ontologies}

    return {ont: weights[ont] / total for ont in active_ontologies}


def _weighted_pick_ontology(
    candidates: List[str], ratio_map: Dict[str, float], rng: random.Random
) -> str:
    """Pick one ontology from candidates using ratio_map weights."""
    if len(candidates) == 1:
        return candidates[0]

    weights = [max(float(ratio_map.get(ont, 0.0)), 0.0) for ont in candidates]
    total = sum(weights)
    if total <= 0:
        weights = [1.0 for _ in candidates]
        total = float(len(candidates))

    pick = rng.random() * total
    running = 0.0
    for ont, w in zip(candidates, weights):
        running += w
        if pick <= running:
            return ont
    return candidates[-1]


def _triplet_batch_size(batch_triplet: Tuple[Batch, Batch, Batch]) -> int:
    ba, _, _ = batch_triplet
    return int(getattr(ba, "num_graphs", 0)) or 0


def _mean_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        metric_name: sum(values) / len(values)
        for metric_name, values in metric_lists.items()
        if values
    }


def _compute_go_phase0_loss(
    model,
    batch_dict: Dict[str, Tuple[Batch, Batch, Batch]],
    device: torch.device,
    margin: float,
    ontologies: List[str],
):
    """
    Pseudocode-style GO loss: average triplet losses over ontologies present
    in this step (each uses its own head via forward_go_head).
    """
    losses = []
    per_ont = {}

    for ont in ontologies:
        triplet = batch_dict.get(ont)
        if triplet is None:
            continue
        ba, bpos, bneg = triplet
        ba = ba.to(device)
        bpos = bpos.to(device)
        bneg = bneg.to(device)

        za = model.forward_go_head(ba, ont)
        zp = model.forward_go_head(bpos, ont)
        zn = model.forward_go_head(bneg, ont)

        loss_ont, d_pos, d_neg = _triplet_loss(za, zp, zn, margin=margin)
        detailed_metrics = compute_detailed_metrics(
            za.detach(), zp.detach(), zn.detach()
        )
        losses.append(loss_ont)
        per_ont[ont] = {
            "loss": loss_ont.item(),
            "avg_pos_dist": d_pos.mean().item(),
            "avg_neg_dist": d_neg.mean().item(),
            "dist_margin": (d_neg.mean() - d_pos.mean()).item(),
            "Alignment": detailed_metrics["Alignment"],
            "Uniformity": detailed_metrics["Uniformity"],
            "MRR": detailed_metrics["MRR"],
        }

    if not losses:
        return None, per_ont

    total = sum(losses) / len(losses)
    return total, per_ont


def _go_collate(batch):
    """
    Collate function for GO phase-0 triplets.

    Each item coming from the dataset is a simple (anchor, positive, negative)
    tuple of torch_geometric.data.Data objects. We need to convert each column
    into a separate Batch so the model can process them.
    """
    # Filter out any None entries (in case dataset decides to skip samples)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    anchors, positives, negatives = zip(*batch)
    ba = Batch.from_data_list(list(anchors))
    bp = Batch.from_data_list(list(positives))
    bn = Batch.from_data_list(list(negatives))
    return ba, bp, bn


def _mean_eval_loss_for_loaders(
    model,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    margin: float,
    ontologies: List[str],
    metric_prefix: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Average eval loss and detailed metrics overall and per ontology."""
    model.eval()
    overall_metric_lists = {
        metric_name: [] for metric_name in _GO_SUMMARIZED_METRIC_NAMES
    }
    per_ont_metric_lists: Dict[str, Dict[str, List[float]]] = {}
    with torch.no_grad():
        for ont in ontologies:
            loader = loaders.get(ont)
            if loader is None:
                continue
            ont_metric_lists = {
                metric_name: [] for metric_name in _GO_SUMMARIZED_METRIC_NAMES
            }
            for batch in loader:
                if batch is None:
                    continue
                batch_dict = {ont: batch}
                loss, per_ont = _compute_go_phase0_loss(
                    model, batch_dict, device, margin, [ont]
                )
                if loss is None or ont not in per_ont:
                    continue
                ont_stats = per_ont[ont]
                for metric_name in _GO_SUMMARIZED_METRIC_NAMES:
                    metric_value = float(ont_stats[metric_name])
                    overall_metric_lists[metric_name].append(metric_value)
                    ont_metric_lists[metric_name].append(metric_value)
                if metric_prefix:
                    batch_log = {
                        f"{metric_prefix}/batch_loss": ont_stats["loss"],
                        f"{metric_prefix}/{ont}/batch_loss": ont_stats["loss"],
                    }
                    for metric_name in _GO_DETAIL_METRIC_NAMES:
                        batch_log[f"{metric_prefix}/{metric_name}_batch"] = (
                            ont_stats[metric_name]
                        )
                        batch_log[f"{metric_prefix}/{ont}/{metric_name}_batch"] = (
                            ont_stats[metric_name]
                        )
                    wandb.log(batch_log)
            if ont_metric_lists["loss"]:
                per_ont_metric_lists[ont] = ont_metric_lists
    return _mean_metric_lists(overall_metric_lists), {
        ont: _mean_metric_lists(metric_lists)
        for ont, metric_lists in per_ont_metric_lists.items()
    }


def _build_go_loader(
    tsv_path: str,
    ontology: str,
    cfg: ProjectConfig,
    prebuilt_graph_root: str,
    shuffle: bool,
    phase0_split: Optional[str] = None,
    protein_to_split: Optional[dict] = None,
) -> Optional[DataLoader]:
    dataset = GOSemanticTripletDataset(
        tsv_path=tsv_path,
        ontology=ontology,
        config=cfg,
        prebuilt_graph_root=prebuilt_graph_root,
        phase0_split=phase0_split,
        protein_to_split=protein_to_split,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=cfg.go_batch_size,
        shuffle=shuffle,
        collate_fn=_go_collate,
        num_workers=getattr(cfg, "go_num_workers", 0),
    )
    return loader


def run_go_pretraining(model, cfg: ProjectConfig, device: torch.device):
    """
    Phase-0 GO semantic pretraining.

    Uses semantic similarity triplets to train MF/BP/CC heads on top of the
    shared encoder.
    """
    if cfg.go_phase0_epochs <= 0:
        return

    print("\n=== Phase 0: GO Semantic Similarity Pretraining ===")

    prebuilt_graph_root = getattr(cfg, "go_prebuilt_graph_root", None)
    if not prebuilt_graph_root or not os.path.isdir(prebuilt_graph_root):
        raise FileNotFoundError(
            "Phase 0 requires go_prebuilt_graph_root to be set to a directory of prebuilt "
            f"PyG graph .pt files (got {prebuilt_graph_root!r})."
        )
    print(f"[Phase0] Prebuilt GO graphs: {prebuilt_graph_root}")

    n_prebuilt = GOSemanticTripletDataset.warm_prebuilt_index(prebuilt_graph_root)
    if n_prebuilt == 0:
        print(
            "[Phase0] No .pt files indexed under prebuilt_graph_root (empty folder or Drive I/O issue). "
            "Skipping Phase 0 — no TSV parsing."
        )
        return

    # Resolve TSV paths
    tsv_dir = cfg.go_tsv_dir
    mf_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv"
    )
    bp_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv"
    )
    cc_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv"
    )

    split_json = getattr(cfg, "go_protein_split_json_path", None)
    if not split_json:
        raise ValueError("Phase 0 requires go_protein_split_json_path (protein_to_split JSON).")
    protein_to_split = load_protein_to_split_json(split_json)
    n_train = sum(1 for s in protein_to_split.values() if s == "train")
    n_val = sum(1 for s in protein_to_split.values() if s == "val")
    n_test = sum(1 for s in protein_to_split.values() if s == "test")
    print(
        f"[Phase0] protein_to_split from {split_json}: "
        f"total={len(protein_to_split):,} | train={n_train:,} val={n_val:,} test={n_test:,}"
    )

    def make_loaders_for_split(split_name: str, shuffle: bool) -> Dict[str, DataLoader]:
        out: Dict[str, DataLoader] = {}
        for ont, path in [("mf", mf_tsv), ("bp", bp_tsv), ("cc", cc_tsv)]:
            if not os.path.exists(path):
                continue
            loader = _build_go_loader(
                tsv_path=path,
                ontology=ont,
                cfg=cfg,
                prebuilt_graph_root=prebuilt_graph_root,
                shuffle=shuffle,
                phase0_split=split_name,
                protein_to_split=protein_to_split,
            )
            if loader is not None:
                out[ont] = loader
        return out

    train_loaders = make_loaders_for_split("train", shuffle=True)
    val_loaders = make_loaders_for_split("val", shuffle=False)
    test_loaders = make_loaders_for_split("test", shuffle=False)
    loaders = train_loaders
    for split_label, ld in (
        ("train", train_loaders),
        ("val", val_loaders),
        ("test", test_loaders),
    ):
        for ont in _GO_ONTOLOGY_ORDER:
            if ont not in ld:
                continue
            n = len(ld[ont].dataset)
            print(f"[Phase0] {split_label} triplets [{ont}]: {n:,}")

    if not loaders:
        print("No GO loaders constructed for phase 0, skipping.")
        return

    # Stable order for averaging (only ontologies that have a loader).
    active_ontologies = [o for o in _GO_ONTOLOGY_ORDER if o in loaders]
    best_checkpoint_path = getattr(cfg, "go_phase0_best_model_path", None)
    last_checkpoint_path = getattr(cfg, "go_phase0_last_model_path", None)
    best_metric = float("inf")
    best_epoch = None
    train_metric_prefix = "phase0-train"
    val_metric_prefix = "phase0-val"
    test_metric_prefix = "phase0-test"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.go_lr, weight_decay=cfg.weight_decay
    )

    model.to(device)

    sampling_enabled = bool(getattr(cfg, "go_sampling_enabled", False))
    ratio_map = _normalize_sampling_ratio(
        getattr(cfg, "go_sampling_ratio", None), active_ontologies
    )
    log_sampling_stats = bool(getattr(cfg, "go_log_sampling_stats", True))
    if sampling_enabled:
        ratio_txt = ", ".join(
            f"{ont}:{ratio_map.get(ont, 0.0):.2f}" for ont in active_ontologies
        )
        print(f"[Phase0] Sampling mode enabled | ratios={ratio_txt}")
    if best_checkpoint_path:
        print(f"[Phase0] Best checkpoint path: {best_checkpoint_path}")
    if last_checkpoint_path:
        print(f"[Phase0] Last checkpoint path: {last_checkpoint_path}")

    for epoch in range(cfg.go_phase0_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_ont_loss_sums = {ont: 0.0 for ont in active_ontologies}
        epoch_ont_loss_counts = {ont: 0 for ont in active_ontologies}
        epoch_metric_sums = {
            metric_name: 0.0 for metric_name in _GO_DETAIL_METRIC_NAMES
        }
        epoch_ont_metric_sums = {
            ont: {metric_name: 0.0 for metric_name in _GO_DETAIL_METRIC_NAMES}
            for ont in active_ontologies
        }

        gens: Dict[str, Iterator] = {
            ont: _infinite_batches(loaders[ont]) for ont in active_ontologies
        }
        if sampling_enabled:
            # Keep epoch throughput comparable to previous loop:
            # previously each step consumed all ontologies.
            n_steps = sum(len(loaders[ont]) for ont in active_ontologies)
        else:
            # Legacy behavior: one batch per ontology per step.
            n_steps = max(len(loaders[ont]) for ont in active_ontologies)
        rng = random.Random(int(getattr(cfg, "go_split_seed", 42)) + epoch)
        sampled_step_counts = {ont: 0 for ont in active_ontologies}
        sampled_batch_counts = {ont: 0 for ont in active_ontologies}

        pbar = tqdm(
            range(n_steps),
            desc=f"Phase0 Epoch {epoch+1}",
            leave=False,
        )
        for _ in pbar:
            batch_dict = {}
            if sampling_enabled:
                tried = set()
                while len(tried) < len(active_ontologies):
                    remaining = [o for o in active_ontologies if o not in tried]
                    ont = _weighted_pick_ontology(remaining, ratio_map, rng)
                    tried.add(ont)
                    b = next(gens[ont])
                    if b is None:
                        continue
                    batch_dict[ont] = b
                    sampled_step_counts[ont] += 1
                    sampled_batch_counts[ont] += _triplet_batch_size(b)
                    break
            else:
                for ont in active_ontologies:
                    b = next(gens[ont])
                    if b is not None:
                        batch_dict[ont] = b
                        sampled_step_counts[ont] += 1
                        sampled_batch_counts[ont] += _triplet_batch_size(b)

            loss, per_ont = _compute_go_phase0_loss(
                model,
                batch_dict,
                device,
                cfg.go_margin,
                active_ontologies,
            )
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            log_dict = {
                f"{train_metric_prefix}/batch_loss": loss.item(),
                f"{train_metric_prefix}/combined_batch_loss": loss.item(),
                f"{train_metric_prefix}/n_ontologies_in_batch": len(per_ont),
            }
            for metric_name in _GO_DETAIL_METRIC_NAMES:
                metric_values = [st[metric_name] for st in per_ont.values()]
                if metric_values:
                    batch_metric_value = sum(metric_values) / len(metric_values)
                    log_dict[f"{train_metric_prefix}/{metric_name}_batch"] = (
                        batch_metric_value
                    )
                    epoch_metric_sums[metric_name] += batch_metric_value
            for ont, st in per_ont.items():
                log_dict[f"{train_metric_prefix}/{ont}/batch_loss"] = st["loss"]
                log_dict[f"{train_metric_prefix}/{ont}/avg_pos_dist"] = st["avg_pos_dist"]
                log_dict[f"{train_metric_prefix}/{ont}/avg_neg_dist"] = st["avg_neg_dist"]
                log_dict[f"{train_metric_prefix}/{ont}/dist_margin"] = st["dist_margin"]
                for metric_name in _GO_DETAIL_METRIC_NAMES:
                    log_dict[f"{train_metric_prefix}/{ont}/{metric_name}_batch"] = (
                        st[metric_name]
                    )
                    epoch_ont_metric_sums[ont][metric_name] += st[metric_name]
                epoch_ont_loss_sums[ont] += st["loss"]
                epoch_ont_loss_counts[ont] += 1
            wandb.log(log_dict)

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
        else:
            avg_loss = 0.0

        train_epoch_log = {
            f"{train_metric_prefix}/loss_epoch": avg_loss,
        }
        for metric_name in _GO_DETAIL_METRIC_NAMES:
            if epoch_steps > 0:
                train_epoch_log[f"{train_metric_prefix}/{metric_name}_epoch"] = (
                    epoch_metric_sums[metric_name] / epoch_steps
                )
        for ont in active_ontologies:
            if epoch_ont_loss_counts[ont] > 0:
                train_epoch_log[f"{train_metric_prefix}/{ont}/loss_epoch"] = (
                    epoch_ont_loss_sums[ont] / epoch_ont_loss_counts[ont]
                )
                for metric_name in _GO_DETAIL_METRIC_NAMES:
                    train_epoch_log[f"{train_metric_prefix}/{ont}/{metric_name}_epoch"] = (
                        epoch_ont_metric_sums[ont][metric_name]
                        / epoch_ont_loss_counts[ont]
                    )
        wandb.log(train_epoch_log)
        if log_sampling_stats:
            total_sampled_steps = sum(sampled_step_counts.values())
            total_sampled_batches = sum(sampled_batch_counts.values())
            sampling_log = {}
            for ont in active_ontologies:
                sampling_log[f"{train_metric_prefix}/sampling/steps_{ont}"] = sampled_step_counts[ont]
                sampling_log[f"{train_metric_prefix}/sampling/samples_{ont}"] = sampled_batch_counts[ont]
                sampling_log[f"{train_metric_prefix}/sampling/target_ratio_{ont}"] = ratio_map.get(
                    ont, 0.0
                )
                sampling_log[f"{train_metric_prefix}/sampling/actual_step_ratio_{ont}"] = (
                    (sampled_step_counts[ont] / total_sampled_steps)
                    if total_sampled_steps > 0
                    else 0.0
                )
                sampling_log[f"{train_metric_prefix}/sampling/actual_sample_ratio_{ont}"] = (
                    (sampled_batch_counts[ont] / total_sampled_batches)
                    if total_sampled_batches > 0
                    else 0.0
                )
            wandb.log(sampling_log)
        print(
            f"[Phase0] Epoch {epoch+1}/{cfg.go_phase0_epochs} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Train MRR: {train_epoch_log.get(f'{train_metric_prefix}/MRR_epoch', 0.0):.4f}"
        )

        v_loss = None
        if protein_to_split and val_loaders:
            v_metrics, v_metrics_per_ont = _mean_eval_loss_for_loaders(
                model,
                val_loaders,
                device,
                cfg.go_margin,
                active_ontologies,
                metric_prefix=val_metric_prefix,
            )
            v_loss = v_metrics.get("loss")
            if v_loss is not None:
                val_log = {f"{val_metric_prefix}/loss_epoch": v_loss}
                for metric_name in _GO_DETAIL_METRIC_NAMES:
                    if metric_name in v_metrics:
                        val_log[f"{val_metric_prefix}/{metric_name}_epoch"] = v_metrics[
                            metric_name
                        ]
                for ont, ont_metrics in v_metrics_per_ont.items():
                    if "loss" in ont_metrics:
                        val_log[f"{val_metric_prefix}/{ont}/loss_epoch"] = ont_metrics[
                            "loss"
                        ]
                    for metric_name in _GO_DETAIL_METRIC_NAMES:
                        if metric_name in ont_metrics:
                            val_log[f"{val_metric_prefix}/{ont}/{metric_name}_epoch"] = (
                                ont_metrics[metric_name]
                            )
                wandb.log(val_log)
                print(
                    f"[Phase0] Val epoch loss: {v_loss:.4f} | "
                    f"Val MRR: {v_metrics.get('MRR', 0.0):.4f}"
                )

        if protein_to_split and test_loaders:
            t_metrics, t_metrics_per_ont = _mean_eval_loss_for_loaders(
                model,
                test_loaders,
                device,
                cfg.go_margin,
                active_ontologies,
                metric_prefix=test_metric_prefix,
            )
            t_loss = t_metrics.get("loss")
            if t_loss is not None:
                test_log = {f"{test_metric_prefix}/loss_epoch": t_loss}
                for metric_name in _GO_DETAIL_METRIC_NAMES:
                    if metric_name in t_metrics:
                        test_log[f"{test_metric_prefix}/{metric_name}_epoch"] = t_metrics[
                            metric_name
                        ]
                for ont, ont_metrics in t_metrics_per_ont.items():
                    if "loss" in ont_metrics:
                        test_log[f"{test_metric_prefix}/{ont}/loss_epoch"] = ont_metrics[
                            "loss"
                        ]
                    for metric_name in _GO_DETAIL_METRIC_NAMES:
                        if metric_name in ont_metrics:
                            test_log[f"{test_metric_prefix}/{ont}/{metric_name}_epoch"] = (
                                ont_metrics[metric_name]
                            )
                wandb.log(test_log)
                print(
                    f"[Phase0] Test epoch loss: {t_loss:.4f} | "
                    f"Test MRR: {t_metrics.get('MRR', 0.0):.4f}"
                )

        selection_metric = v_loss if v_loss is not None else avg_loss
        selection_metric_name = "val_loss_epoch" if v_loss is not None else "train_loss_epoch"
        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch + 1
            _save_model_checkpoint(model, best_checkpoint_path)
            if best_checkpoint_path:
                print(
                    f"[Phase0] Saved best checkpoint ({selection_metric_name}="
                    f"{selection_metric:.4f}) to {best_checkpoint_path}"
                )
            else:
                print(
                    f"[Phase0] Updated best {selection_metric_name}: "
                    f"{selection_metric:.4f}"
                )

        wandb.log(
            {
                f"{train_metric_prefix}/best_epoch_so_far": best_epoch if best_epoch is not None else 0,
                f"{train_metric_prefix}/best_metric_so_far": best_metric,
                f"{train_metric_prefix}/selection_metric": selection_metric,
            }
        )

    _save_model_checkpoint(model, last_checkpoint_path)
    if last_checkpoint_path:
        print(f"[Phase0] Saved last checkpoint to {last_checkpoint_path}")
