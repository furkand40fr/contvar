"""
Variant-Specific GO Benchmark Evaluation.

Evaluates decoder models on the LOF/GOF benchmark dataset.
For each benchmark entry (protein_id, matched_go_id, label):
  - LOF: WT protein SHOULD have this GO term → decoder score should be HIGH
  - GOF: WT protein should NOT have this GO term → decoder score should be LOW

This tests the decoder's ability to correctly identify WT protein functions,
which is a prerequisite for detecting LOF/GOF from variant embeddings.

Usage:
    python -m phase2_decoder.benchmark_eval
    python -m phase2_decoder.benchmark_eval --aspect F --embedding esm
    python -m phase2_decoder.benchmark_eval --aspect F --embedding contvar
    python -m phase2_decoder.benchmark_eval --aspect F --embedding concat
    python -m phase2_decoder.benchmark_eval --all
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from phase2_decoder.models.ffn_decoder import FFNDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark evaluation run."""
    aspect: str                          # "F", "P", or "C"
    embedding_type: str                  # "esm", "contvar", or "concat"
    benchmark_tsv: str = "variant_specific_go_benchmark.tsv"
    esm_h5: str = "esm2_t33_650M_UR50D_protein_embedding.h5"
    contvar_h5: str = "go_pretraining_contvar_embeddings.h5"
    uniref_tsv: str = "protein_uniref50.tsv"
    split_json: str = "phase0_go_split.json"
    goa_tsv: str = "goa_2025-12-04_swissprot_noiea.tsv"
    min_go_freq: int = 10

    # Checkpoint and architecture — auto-detected or manually specified
    checkpoint: Optional[str] = None
    input_dim: Optional[int] = None
    hidden_dims: Optional[list] = None
    dropout: float = 0.3

    def __post_init__(self):
        asp = self.aspect.lower()
        if self.checkpoint is None:
            self.checkpoint = f"decoder_best_{asp}.pt"
        self.go_vocab_json = f"go_vocab_{asp}.json"


# ── Embedding Types ──────────────────────────────────────────────────────────

EMBEDDING_CONFIGS = {
    "esm": {
        "label": "ESM-2 (1280d)",
        "input_dim": 1280,
        "hidden_dims": [512, 1024, 512],
    },
    "contvar": {
        "label": "ContVAR Phase1 (256d)",
        "input_dim": 256,
        "hidden_dims": [256, 256],
    },
    "concat": {
        "label": "ESM + ContVAR (1536d)",
        "input_dim": 1536,
        "hidden_dims": [1024, 512],
    },
}


# ── Split Loading ────────────────────────────────────────────────────────────

def load_protein_splits(uniref_tsv: str, split_json: str) -> dict:
    """Returns protein_id → split mapping."""
    with open(split_json) as f:
        phase0 = json.load(f)
    group_to_split = phase0["group_to_split"]

    protein_to_split = {}
    with open(uniref_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            grp = row["group_id"]
            split = group_to_split.get(grp)
            if split is not None:
                protein_to_split[row["protein_id"]] = split
    return protein_to_split


# ── Model Loading ────────────────────────────────────────────────────────────

def auto_detect_architecture(checkpoint_path: str) -> dict:
    """Auto-detect model architecture from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # Find input dim from first linear layer
    first_weight = sd.get("net.0.weight")
    if first_weight is None:
        raise ValueError("Cannot find net.0.weight in checkpoint")
    input_dim = first_weight.shape[1]

    # Find all hidden dims (linear layers before output)
    linear_keys = sorted(
        [k for k in sd if k.endswith(".weight") and k.startswith("net.") and len(sd[k].shape) == 2]
    )
    # Last one is output layer
    hidden_dims = [sd[k].shape[0] for k in linear_keys[:-1]]
    output_dim = sd[linear_keys[-1]].shape[0]

    has_rescaling = "input_scale" in sd

    return {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "has_rescaling": has_rescaling,
        "state_dict": sd,
    }


def load_decoder_model(checkpoint_path: str, n_classes: int,
                       input_dim: int = None, hidden_dims: list = None,
                       dropout: float = 0.3) -> torch.nn.Module:
    """Load decoder model, auto-detecting architecture if not specified."""
    arch = auto_detect_architecture(checkpoint_path)
    sd = arch["state_dict"]

    actual_input = input_dim or arch["input_dim"]
    actual_hidden = hidden_dims or arch["hidden_dims"]

    decoder = FFNDecoder(actual_input, n_classes, actual_hidden, dropout)

    # Handle learnable rescaling parameters
    if arch["has_rescaling"]:
        decoder.input_scale = torch.nn.Parameter(torch.ones(actual_input))
        decoder.input_bias = torch.nn.Parameter(torch.zeros(actual_input))
        # Override forward to apply rescaling
        original_forward = decoder.forward

        def rescaled_forward(x):
            x = x * decoder.input_scale + decoder.input_bias
            return decoder.net(x)

        decoder.forward = rescaled_forward

    decoder.load_state_dict(sd, strict=False)
    decoder.eval().to(DEVICE)
    return decoder


# ── Embedding Loading ────────────────────────────────────────────────────────

def load_wt_embeddings(protein_ids: list, embedding_type: str,
                       esm_h5: str, contvar_h5: str) -> dict:
    """
    Load WT protein embeddings for the given protein IDs.
    Returns: {protein_id: tensor[D]}
    """
    embeddings = {}

    if embedding_type == "esm":
        with h5py.File(esm_h5, "r") as h5f:
            available = set(h5f.keys())
            for pid in protein_ids:
                if pid in available:
                    embeddings[pid] = torch.FloatTensor(h5f[pid][:])

    elif embedding_type == "contvar":
        with h5py.File(contvar_h5, "r") as h5f:
            available = set(h5f.keys())
            for pid in protein_ids:
                if pid in available:
                    embeddings[pid] = torch.FloatTensor(h5f[pid][:])

    elif embedding_type == "concat":
        with h5py.File(esm_h5, "r") as h5_esm, \
             h5py.File(contvar_h5, "r") as h5_cv:
            esm_keys = set(h5_esm.keys())
            cv_keys = set(h5_cv.keys())
            common = esm_keys & cv_keys
            for pid in protein_ids:
                if pid in common:
                    esm_emb = torch.FloatTensor(h5_esm[pid][:])
                    cv_emb = torch.FloatTensor(h5_cv[pid][:])
                    embeddings[pid] = torch.cat([esm_emb, cv_emb])

    return embeddings


# ── Main Evaluation ──────────────────────────────────────────────────────────

def evaluate_benchmark(cfg: BenchmarkConfig) -> dict:
    """
    Run benchmark evaluation for a single configuration.
    Returns metrics dict.
    """
    asp = cfg.aspect.lower()
    emb_label = EMBEDDING_CONFIGS[cfg.embedding_type]["label"]
    print(f"\n{'='*70}")
    print(f"  Benchmark Evaluation: {emb_label} | Aspect={cfg.aspect}")
    print(f"{'='*70}")

    # 1. Load GO vocabulary
    if not Path(cfg.go_vocab_json).exists():
        print(f"  Building GO vocab for aspect {cfg.aspect}...")
        from phase2_decoder.data.parse_goa import parse_goa_tsv, build_go_vocab
        annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect)
        go_vocab = build_go_vocab(annotations, min_freq=cfg.min_go_freq,
                                  save_path=cfg.go_vocab_json)
    else:
        with open(cfg.go_vocab_json) as f:
            go_vocab = json.load(f)
    n_classes = len(go_vocab)
    vocab_gos = set(go_vocab.keys()) - {"NULL_FUNCTION"}
    print(f"  GO vocab: {n_classes} terms (aspect {cfg.aspect})")

    # 2. Load and verify checkpoint architecture
    print(f"  Checkpoint: {cfg.checkpoint}")
    arch = auto_detect_architecture(cfg.checkpoint)
    expected_dim = EMBEDDING_CONFIGS[cfg.embedding_type]["input_dim"]

    if arch["input_dim"] != expected_dim:
        print(f"  [SKIP] checkpoint input_dim={arch['input_dim']} "
              f"doesn't match {cfg.embedding_type} expected_dim={expected_dim}")
        return None

    # 3. Load decoder
    decoder = load_decoder_model(
        cfg.checkpoint, n_classes,
        input_dim=expected_dim,
        hidden_dims=arch["hidden_dims"],
        dropout=cfg.dropout,
    )
    print(f"  Decoder loaded: input={expected_dim}, hidden={arch['hidden_dims']}, "
          f"output={arch['output_dim']}")

    # 4. Load benchmark data
    benchmark = pd.read_csv(cfg.benchmark_tsv, sep="\t")
    benchmark = benchmark[benchmark["matched_go_id"].isin(vocab_gos)]
    print(f"  Benchmark rows with GO in vocab: {len(benchmark)}")

    if len(benchmark) == 0:
        print("  [SKIP] No benchmark rows match this aspect's GO vocab!")
        return None

    # 5. Load splits
    protein_to_split = load_protein_splits(cfg.uniref_tsv, cfg.split_json)
    benchmark["split"] = benchmark["protein_id"].map(protein_to_split)

    # 6. Load WT embeddings for all benchmark proteins
    all_proteins = benchmark["protein_id"].unique().tolist()
    print(f"  Loading embeddings for {len(all_proteins)} proteins...")
    embeddings = load_wt_embeddings(
        all_proteins, cfg.embedding_type,
        cfg.esm_h5, cfg.contvar_h5,
    )
    print(f"  Embeddings loaded: {len(embeddings)}/{len(all_proteins)} proteins")

    # Filter benchmark to proteins with embeddings
    benchmark = benchmark[benchmark["protein_id"].isin(embeddings)]
    print(f"  Benchmark rows with embeddings: {len(benchmark)}")

    # 7. Run decoder predictions (batch by protein for efficiency)
    print("  Running decoder predictions...")
    protein_scores = {}  # {protein_id: scores_array[n_classes]}

    with torch.no_grad():
        unique_proteins = benchmark["protein_id"].unique()
        batch_size = 256
        for i in range(0, len(unique_proteins), batch_size):
            batch_pids = unique_proteins[i:i+batch_size]
            batch_embs = torch.stack([embeddings[pid] for pid in batch_pids])
            batch_logits = decoder(batch_embs.to(DEVICE))
            batch_probs = torch.sigmoid(batch_logits).cpu().numpy()
            for j, pid in enumerate(batch_pids):
                protein_scores[pid] = batch_probs[j]

    # 8. Evaluate per split
    results = {}
    for split_name in ["val", "test"]:
        split_df = benchmark[benchmark["split"] == split_name].copy()
        if len(split_df) == 0:
            continue

        scores = []
        labels = []  # 1 = LOF (expect high score), 0 = GOF (expect low score)

        for _, row in split_df.iterrows():
            pid = row["protein_id"]
            go_id = row["matched_go_id"]
            label = row["label"]

            go_idx = go_vocab[go_id]
            score = protein_scores[pid][go_idx]

            scores.append(score)
            # For LOF: we expect score to be HIGH (WT has this function)
            # For GOF: we expect score to be LOW (WT lacks this function)
            labels.append(1 if label == "LOF" else 0)

        scores = np.array(scores)
        labels = np.array(labels)

        n_lof = int(labels.sum())
        n_gof = int(len(labels) - labels.sum())

        # Threshold sweep
        best_acc = 0
        best_thr = 0.5
        for thr in np.arange(0.05, 0.96, 0.05):
            preds = (scores >= thr).astype(int)
            acc = (preds == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr

        preds = (scores >= best_thr).astype(int)

        # LOF metrics (label=1, expect score >= thr)
        lof_mask = labels == 1
        lof_recall = preds[lof_mask].mean() if lof_mask.sum() > 0 else 0.0

        # GOF metrics (label=0, expect score < thr)
        gof_mask = labels == 0
        gof_recall = (1 - preds[gof_mask]).mean() if gof_mask.sum() > 0 else 0.0

        overall_acc = (preds == labels).mean()

        # AUROC and AUPRC
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(labels, scores)
        except ValueError:
            auprc = float("nan")

        # Precision, Recall, F1 (treating LOF=positive)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )

        split_results = {
            "split": split_name,
            "n_total": len(labels),
            "n_lof": n_lof,
            "n_gof": n_gof,
            "threshold": round(best_thr, 2),
            "overall_acc": round(overall_acc, 4),
            "lof_recall": round(lof_recall, 4),
            "gof_recall": round(gof_recall, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        }
        results[split_name] = split_results

        print(f"\n  [{split_name.upper()}] n={len(labels)} (LOF={n_lof}, GOF={n_gof})")
        print(f"    Threshold: {best_thr:.2f}")
        print(f"    Overall Accuracy:  {overall_acc:.4f}")
        print(f"    LOF Recall:        {lof_recall:.4f}")
        print(f"    GOF Recall:        {gof_recall:.4f}")
        print(f"    Precision/Recall:  {prec:.4f} / {rec:.4f}")
        print(f"    F1:                {f1:.4f}")
        print(f"    AUROC:             {auroc:.4f}")
        print(f"    AUPRC:             {auprc:.4f}")

    return {
        "aspect": cfg.aspect,
        "embedding": emb_label,
        "embedding_type": cfg.embedding_type,
        "results": results,
    }


# ── Summary Table ────────────────────────────────────────────────────────────

def print_summary_table(all_results: list):
    """Print a formatted summary table of all evaluations."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        for split_name in ["val", "test"]:
            if split_name not in res["results"]:
                continue
            sr = res["results"][split_name]
            rows.append({
                "Embedding": res["embedding"],
                "Aspect": res["aspect"],
                "Split": split_name,
                "N": sr["n_total"],
                "LOF": sr["n_lof"],
                "GOF": sr["n_gof"],
                "Thr": sr["threshold"],
                "Acc": sr["overall_acc"],
                "LOF_Rec": sr["lof_recall"],
                "GOF_Rec": sr["gof_recall"],
                "F1": sr["f1"],
                "AUROC": sr["auroc"],
                "AUPRC": sr["auprc"],
            })

    if not rows:
        print("\nNo results to display.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  BENCHMARK EVALUATION SUMMARY -- LOF/GOF Prediction")
    print("=" * 90)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print()

    return df


def save_results_csv(all_results: list, output_path: str = "benchmark_results.csv"):
    """Save detailed results to CSV."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        for split_name in ["val", "test"]:
            if split_name not in res["results"]:
                continue
            sr = res["results"][split_name]
            rows.append({
                "embedding": res["embedding"],
                "embedding_type": res["embedding_type"],
                "aspect": res["aspect"],
                "split": split_name,
                **sr,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate decoder on variant-specific GO benchmark"
    )
    parser.add_argument("--aspect", default=None, choices=["F", "P", "C"],
                        help="GO aspect (default: all)")
    parser.add_argument("--embedding", default=None,
                        choices=["esm", "contvar", "concat"],
                        help="Embedding type (default: all)")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path")
    parser.add_argument("--all", action="store_true",
                        help="Run all valid aspect × embedding combinations")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    aspects = [args.aspect] if args.aspect else ["F", "P", "C"]
    emb_types = [args.embedding] if args.embedding else ["esm", "contvar", "concat"]

    all_results = []
    for aspect in aspects:
        for emb_type in emb_types:
            cfg = BenchmarkConfig(
                aspect=aspect,
                embedding_type=emb_type,
            )
            if args.checkpoint:
                cfg.checkpoint = args.checkpoint

            # Check if checkpoint exists
            if not Path(cfg.checkpoint).exists():
                print(f"\n[WARN] Checkpoint not found: {cfg.checkpoint} -- skipping {emb_type}/{aspect}")
                continue

            result = evaluate_benchmark(cfg)
            if result is not None:
                all_results.append(result)

    # Summary
    df = print_summary_table(all_results)
    save_results_csv(all_results, args.output)


if __name__ == "__main__":
    main()
