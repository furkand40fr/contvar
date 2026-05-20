"""
Batch variant function change prediction.

For every variant in embeddings_variable.h5:
    1. Parse the WT UniProt ID from the key
    2. Retrieve WT GO annotations from GOA
    3. Mean-pool the variant embedding: (L, 1280) → (1280,)
    4. Run predict_function_change()
    5. Write results to CSV

Usage:
    python -m phase2_decoder.predict_variants
    python -m phase2_decoder.predict_variants --var_h5 embeddings_variable.h5 --out predictions.csv
"""

import argparse
import csv
import re
import h5py
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from phase2_decoder.config import DecoderConfig
from phase2_decoder.data.parse_goa import parse_goa_tsv
from phase2_decoder.inference import load_decoder, predict_function_change


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UniProt accession regex with optional isoform suffix.
_UNIPROT_RE = re.compile(
    r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$'
)
_MUTATION_RE = re.compile(r'^[A-Z]\d+[A-Z*]$', re.IGNORECASE)


def mean_pool(h5f, key: str) -> torch.Tensor:
    """
    Per-residue embedding (L, 1280) → global embedding (1280,) via mean pooling.
    Returns as-is if already (1280,).
    """
    emb = torch.FloatTensor(h5f[key][:])
    if emb.dim() == 2:
        emb = emb.mean(dim=0)
    return emb


def parse_wt_id(variant_key: str, goa_ids: set) -> str | None:
    """
    Finds all UniProt accession candidates in the key and returns
    the first one present in GOA. Returns None if not found.

    Examples:
      'A0A1I9GEU1_NEIME_..._A0A1I9GEU1_A109V_1' → 'A0A1I9GEU1' (if in GOA)
      'ADRB2_HUMAN_Jones_2020_P07550_A119M_0'    → 'P07550'
      'BLAT_ECOLX_Stiffler_2015_P62593_A124V_1'  → 'P62593'
    """
    normalized_goa_ids = {pid.upper(): pid for pid in goa_ids}
    for part in variant_key.split("_"):
        candidate = part.upper()
        if _UNIPROT_RE.match(candidate) and candidate in normalized_goa_ids:
            return normalized_goa_ids[candidate]
    return None


def run(var_h5: str, out_csv: str, cfg: DecoderConfig):
    # GO annotations
    print("Loading GO annotations...")
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect)

    # Decoder + vocab
    print("Loading decoder...")
    decoder, go_vocab = load_decoder(
        checkpoint=cfg.decoder_checkpoint,
        go_vocab_json=cfg.go_vocab_json,
        encoder_output_dim=cfg.encoder_output_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )

    # wandb başlat
    wandb.login(key=cfg.wandb_api_key)
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"predict-variants-{cfg.aspect.lower()}",
        config={"var_h5": var_h5, "out_csv": out_csv,
                "aspect": cfg.aspect, "checkpoint": cfg.decoder_checkpoint},
    )

    # CSV başlık
    fieldnames = ["variant_id", "wt_id", "exp_label", "ContVAR_fitness",
                  "n_loss", "n_gain", "n_neutral", "wt_go_count"]

    skipped = 0
    all_fitness: list[float] = []
    all_labels:  list[int]   = []

    with h5py.File(var_h5, "r") as h5f, \
         open(out_csv, "w", newline="") as fout:

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        keys = list(h5f.keys())
        print(f"Total variants: {len(keys)}")

        for key in tqdm(keys, desc="Predicting"):
            # Skip WT records
            parts = key.split("_")
            if not any(_MUTATION_RE.match(p) for p in parts):
                skipped += 1
                continue

            wt_id = parse_wt_id(key, set(annotations.keys()))

            if wt_id is None:
                skipped += 1
                continue

            # Experimental label: last part of the key (0 or 1)
            try:
                exp_label = int(parts[-1])
            except ValueError:
                exp_label = -1  # unknown

            wt_go_terms = annotations[wt_id]

            var_emb = mean_pool(h5f, key)
            result  = predict_function_change(wt_go_terms, var_emb, decoder, go_vocab)

            fitness   = result.pop("ContVAR_fitness")
            n_loss    = sum(1 for v in result.values() if v["status"] == "LOSS")
            n_gain    = sum(1 for v in result.values() if v["status"] == "GAIN")
            n_neutral = sum(1 for v in result.values() if v["status"] == "NEUTRAL")

            writer.writerow({
                "variant_id":      key,
                "wt_id":           wt_id,
                "exp_label":       exp_label,
                "ContVAR_fitness": fitness,
                "n_loss":          n_loss,
                "n_gain":          n_gain,
                "n_neutral":       n_neutral,
                "wt_go_count":     len(wt_go_terms),
            })

            if exp_label in (0, 1):
                all_fitness.append(fitness)
                all_labels.append(exp_label)

    # Compute and log AUROC / AUPRC
    print(f"\nDone → {out_csv}")
    print(f"Skipped (no GO annotation): {skipped}")

    if len(all_labels) >= 2 and len(set(all_labels)) == 2:
        fitness_arr = np.array(all_fitness)
        label_arr   = np.array(all_labels)

        auroc = roc_auc_score(label_arr, fitness_arr)
        auprc = average_precision_score(label_arr, fitness_arr)

        print(f"AUROC (fitness vs exp_label): {auroc:.4f}")
        print(f"AUPRC (fitness vs exp_label): {auprc:.4f}")

        wandb.log({
            "n_predicted":                  len(all_fitness),
            "n_skipped":                    skipped,
            "fitness_mean_label0":          float(fitness_arr[label_arr == 0].mean()),
            "fitness_mean_label1":          float(fitness_arr[label_arr == 1].mean()),
            "AUROC_fitness_vs_exp_label":   auroc,
            "AUPRC_fitness_vs_exp_label":   auprc,
        })
        wandb.run.summary["AUROC_fitness_vs_exp_label"] = auroc
        wandb.run.summary["AUPRC_fitness_vs_exp_label"] = auprc
    else:
        wandb.log({"n_predicted": len(all_fitness), "n_skipped": skipped})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var_h5", default="embeddings_variable.h5")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    args = parser.parse_args()

    cfg = DecoderConfig(aspect=args.aspect)
    out = args.out or f"predictions_{args.aspect.lower()}.csv"
    run(args.var_h5, out, cfg)
