"""
Three-stage decoder grid search.

This script does not write to experiment_results.xlsx. It only trains runs,
prints a leaderboard, and writes an optional CSV summary for manual review.

Usage:
    python -m phase2_decoder.grid_search --aspect F --embedding concat
    python -m phase2_decoder.grid_search --aspect all --embedding all --dry-run
    python -m phase2_decoder.grid_search --aspect F --embedding concat --run-final
"""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from phase2_decoder.config import DecoderConfig
from phase2_decoder.train import train


LR_GRID = [2.5e-4, 5e-4, 1e-3]
DROPOUT_GRID = [0.2, 0.3, 0.4]

HIDDEN_DIMS_GRID = [
    [512, 512],
    [1024, 512],
    [512, 1024, 512],
]
WEIGHT_DECAY_GRID = [1e-5, 1e-4, 5e-4]

POS_WEIGHT_CLAMP_GRID = [10.0, 20.0, 40.0]

BASE_LR = 5e-4
BASE_DROPOUT = 0.3
BASE_HIDDEN_DIMS = [512, 1024, 512]
BASE_WEIGHT_DECAY = 1e-4
BASE_POS_WEIGHT_CLAMP = 20.0

CSV_FIELDS = [
    "tag",
    "stage",
    "aspect",
    "embedding_type",
    "lr",
    "dropout",
    "hidden_dims",
    "weight_decay",
    "pos_weight_clamp",
    "epochs",
    "patience",
    "best_epoch",
    "best_val_map",
    "val_mAP",
    "val_F1_macro",
    "val_MCC",
    "val_threshold",
    "test_mAP",
    "test_F1_macro",
    "test_MCC",
    "test_threshold",
    "checkpoint",
]


@dataclass(frozen=True)
class RunParams:
    lr: float = BASE_LR
    dropout: float = BASE_DROPOUT
    hidden_dims: tuple[int, ...] = tuple(BASE_HIDDEN_DIMS)
    weight_decay: float = BASE_WEIGHT_DECAY
    pos_weight_clamp: float = BASE_POS_WEIGHT_CLAMP


def _choices(value: str, all_values: list[str]) -> list[str]:
    return all_values if value == "all" else [value]


def _tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _tag_hidden(hidden_dims: Iterable[int]) -> str:
    return "-".join(str(x) for x in hidden_dims)


def _run_id(stage: str, params: RunParams) -> str:
    return (
        f"{stage}"
        f"_lr{_tag_float(params.lr)}"
        f"_do{_tag_float(params.dropout)}"
        f"_hd{_tag_hidden(params.hidden_dims)}"
        f"_wd{_tag_float(params.weight_decay)}"
        f"_pwc{_tag_float(params.pos_weight_clamp)}"
    )


def _make_cfg(
    aspect: str,
    embedding_type: str,
    params: RunParams,
    stage: str,
    args: argparse.Namespace,
) -> DecoderConfig:
    cfg = DecoderConfig(aspect=aspect, embedding_type=embedding_type)
    cfg.lr = params.lr
    cfg.dropout = params.dropout
    cfg.hidden_dims = list(params.hidden_dims)
    cfg.weight_decay = params.weight_decay
    cfg.pos_weight_clamp = params.pos_weight_clamp
    cfg.epochs = args.epochs if stage != "final" else args.final_epochs
    cfg.early_stop_patience = (
        args.patience if stage != "final" else args.final_patience
    )
    cfg.seed = args.seed
    cfg._return_metrics = True
    cfg.use_go_propagation = args.propagation

    run_id = _run_id(stage, params)
    ckpt_dir = args.checkpoint_dir / args.tag / aspect.lower() / embedding_type
    cfg.decoder_checkpoint = str(ckpt_dir / f"{run_id}.pt")
    cfg.wandb_run_name = f"decoder-grid-{args.tag}-{aspect.lower()}-{embedding_type}-{run_id}"
    return cfg


def _to_row(
    tag: str,
    stage: str,
    result: dict,
) -> dict:
    hp = result["hyperparams"]
    val = result["val"]
    test = result["test"]
    return {
        "tag": tag,
        "stage": stage,
        "aspect": result["aspect"],
        "embedding_type": result["embedding_type"],
        "lr": hp["lr"],
        "dropout": hp["dropout"],
        "hidden_dims": "-".join(str(x) for x in hp["hidden_dims"]),
        "weight_decay": hp["weight_decay"],
        "pos_weight_clamp": hp["pos_weight_clamp"],
        "epochs": hp["epochs"],
        "patience": hp["early_stop_patience"],
        "best_epoch": result["best_epoch"],
        "best_val_map": result["best_val_map"],
        "val_mAP": val["mAP"],
        "val_F1_macro": val["F1_macro"],
        "val_MCC": val["MCC"],
        "val_threshold": val["threshold"],
        "test_mAP": test["mAP"],
        "test_F1_macro": test["F1_macro"],
        "test_MCC": test["MCC"],
        "test_threshold": test["threshold"],
        "checkpoint": result["checkpoint"],
    }


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _print_leaderboard(rows: list[dict], title: str, limit: int = 10):
    # Filter out dry-run rows that have no metrics
    rows = [r for r in rows if r.get("val_mAP", "") != ""]
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    ranked = sorted(rows, key=lambda r: float(r["val_mAP"]), reverse=True)
    for i, row in enumerate(ranked[:limit], 1):
        print(
            f"{i:2d}. val_mAP={float(row['val_mAP']):.4f} "
            f"test_mAP={float(row['test_mAP']):.4f} "
            f"test_MCC={float(row['test_MCC']):.4f} "
            f"aspect={row['aspect']} emb={row['embedding_type']} "
            f"lr={row['lr']} dropout={row['dropout']} "
            f"hidden={row['hidden_dims']} wd={row['weight_decay']} "
            f"pwc={row['pos_weight_clamp']}"
        )


def _run_one(
    aspect: str,
    embedding_type: str,
    params: RunParams,
    stage: str,
    args: argparse.Namespace,
) -> dict:
    run_id = _run_id(stage, params)
    print(f"\n[{aspect}/{embedding_type}] {run_id}")
    if args.dry_run:
        return {
            "tag": args.tag,
            "stage": stage,
            "aspect": aspect,
            "embedding_type": embedding_type,
            "lr": params.lr,
            "dropout": params.dropout,
            "hidden_dims": _tag_hidden(params.hidden_dims),
            "weight_decay": params.weight_decay,
            "pos_weight_clamp": params.pos_weight_clamp,
            "epochs": args.epochs if stage != "final" else args.final_epochs,
            "patience": args.patience if stage != "final" else args.final_patience,
            "best_epoch": "",
            "best_val_map": "",
            "val_mAP": "",
            "val_F1_macro": "",
            "val_MCC": "",
            "val_threshold": "",
            "test_mAP": "",
            "test_F1_macro": "",
            "test_MCC": "",
            "test_threshold": "",
            "checkpoint": "",
        }

    cfg = _make_cfg(aspect, embedding_type, params, stage, args)
    Path(cfg.decoder_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    result = train(cfg)
    row = _to_row(args.tag, stage, result)
    if args.output:
        _append_csv(args.output, row)
    return row


def run_grid_for_pair(aspect: str, embedding_type: str, args: argparse.Namespace) -> list[dict]:
    rows = []

    stage1_rows = []
    for lr in LR_GRID:
        for dropout in DROPOUT_GRID:
            params = RunParams(lr=lr, dropout=dropout)
            stage1_rows.append(_run_one(aspect, embedding_type, params, "stage1", args))
    rows.extend(stage1_rows)
    if args.dry_run:
        return rows

    best1 = max(stage1_rows, key=lambda r: float(r["val_mAP"]))
    best1_params = RunParams(
        lr=float(best1["lr"]),
        dropout=float(best1["dropout"]),
    )
    print(
        f"\nBest stage1 for {aspect}/{embedding_type}: "
        f"lr={best1_params.lr}, dropout={best1_params.dropout}, "
        f"val_mAP={float(best1['val_mAP']):.4f}"
    )

    stage2_rows = []
    for hidden_dims in HIDDEN_DIMS_GRID:
        for weight_decay in WEIGHT_DECAY_GRID:
            params = replace(
                best1_params,
                hidden_dims=tuple(hidden_dims),
                weight_decay=weight_decay,
            )
            stage2_rows.append(_run_one(aspect, embedding_type, params, "stage2", args))
    rows.extend(stage2_rows)

    best2 = max(stage2_rows, key=lambda r: float(r["val_mAP"]))
    best2_params = RunParams(
        lr=float(best2["lr"]),
        dropout=float(best2["dropout"]),
        hidden_dims=tuple(int(x) for x in best2["hidden_dims"].split("-")),
        weight_decay=float(best2["weight_decay"]),
    )
    print(
        f"\nBest stage2 for {aspect}/{embedding_type}: "
        f"hidden={best2['hidden_dims']}, wd={best2_params.weight_decay}, "
        f"val_mAP={float(best2['val_mAP']):.4f}"
    )

    stage3_rows = []
    for pos_weight_clamp in POS_WEIGHT_CLAMP_GRID:
        params = replace(best2_params, pos_weight_clamp=pos_weight_clamp)
        stage3_rows.append(_run_one(aspect, embedding_type, params, "stage3", args))
    rows.extend(stage3_rows)

    best3 = max(stage3_rows, key=lambda r: float(r["val_mAP"]))
    best3_params = RunParams(
        lr=float(best3["lr"]),
        dropout=float(best3["dropout"]),
        hidden_dims=tuple(int(x) for x in best3["hidden_dims"].split("-")),
        weight_decay=float(best3["weight_decay"]),
        pos_weight_clamp=float(best3["pos_weight_clamp"]),
    )
    print(
        f"\nBest stage3 for {aspect}/{embedding_type}: "
        f"pwc={best3_params.pos_weight_clamp}, val_mAP={float(best3['val_mAP']):.4f}"
    )

    if args.run_final:
        rows.append(_run_one(aspect, embedding_type, best3_params, "final", args))

    _print_leaderboard(rows, f"Leaderboard {aspect}/{embedding_type}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run decoder hyperparameter grid search.")
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C", "all"])
    parser.add_argument(
        "--embedding",
        default="concat",
        choices=["esm", "contvar", "contvar_full", "concat", "concat_full", "all"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--final-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--propagation", action="store_true",
                        help="Enable GO hierarchy propagation as postprocessing.")
    parser.add_argument(
        "--tag",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run tag used in checkpoint paths and CSV rows.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("decoder_grid_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("decoder_grid_results.csv"),
        help="CSV summary path. Use an empty string to skip CSV writing.",
    )
    args = parser.parse_args()
    if str(args.output) == "":
        args.output = None

    aspects = _choices(args.aspect, ["F", "P", "C"])
    embeddings = _choices(args.embedding, ["esm", "contvar", "contvar_full", "concat", "concat_full"])

    all_rows = []
    for aspect in aspects:
        for embedding_type in embeddings:
            all_rows.extend(run_grid_for_pair(aspect, embedding_type, args))

    if args.dry_run and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDry-run plan written to {args.output}")

    _print_leaderboard(all_rows, "Overall leaderboard")


if __name__ == "__main__":
    main()