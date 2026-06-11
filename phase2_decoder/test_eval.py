"""
Final evaluation on the test set.

Usage:
    python -m phase2_decoder.test_eval
"""

import torch
from torch.utils.data import DataLoader
import wandb

from phase2_decoder.config import DecoderConfig
from phase2_decoder.data.parse_goa import parse_goa_tsv, build_go_vocab
from phase2_decoder.data.dataset import GOAnnotationDataset
from phase2_decoder.models.ffn_decoder import FFNDecoder
from phase2_decoder.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_eval(cfg: DecoderConfig = None):
    if cfg is None:
        cfg = DecoderConfig()

    print(f"Parsing GO annotations (aspect={cfg.aspect})...")
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect)
    go_vocab    = build_go_vocab(annotations, min_freq=cfg.min_go_freq)
    n_classes   = len(go_vocab)
    print(f"GO vocab size: {n_classes}")

    val_ds = GOAnnotationDataset(
        cfg.esm_h5, cfg.contvar_h5, annotations, go_vocab,
        split="val",
        uniref_tsv=cfg.uniref_tsv,
        split_json=cfg.split_json,
        embedding_type=cfg.embedding_type,
    )
    test_ds = GOAnnotationDataset(
        cfg.esm_h5, cfg.contvar_h5, annotations, go_vocab,
        split="test",
        uniref_tsv=cfg.uniref_tsv,
        split_json=cfg.split_json,
        embedding_type=cfg.embedding_type,
    )
    print(f"Validation: {len(val_ds)} | Test: {len(test_ds)} proteins")

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"Loading best model from {cfg.decoder_checkpoint} for final evaluation...")
    decoder = FFNDecoder(cfg.encoder_output_dim, n_classes,
                       cfg.hidden_dims, cfg.dropout).to(DEVICE)
    decoder.load_state_dict(torch.load(cfg.decoder_checkpoint, map_location=DEVICE))
    decoder.eval()

    wandb.login(key=cfg.wandb_api_key)
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"test-eval-{cfg.aspect.lower()}",
        config={
            "checkpoint": cfg.decoder_checkpoint,
            "threshold":  cfg.eval_threshold,
            "val_size":   len(val_ds),
            "test_size":  len(test_ds),
        },
    )

    asp = cfg.aspect.lower()
    
    val_metrics = evaluate(decoder, val_loader, DEVICE, aspect=cfg.aspect, threshold=cfg.eval_threshold)
    test_metrics = evaluate(decoder, test_loader, DEVICE, aspect=cfg.aspect, threshold=cfg.eval_threshold)
    
    import pandas as pd
    metrics_keys = ["mAP", "F1_macro", "Accuracy", "Precision", "Recall", "MCC"]
    
    val_row = {k: val_metrics[f"{k}_{asp}"] for k in metrics_keys}
    test_row = {k: test_metrics[f"{k}_{asp}"] for k in metrics_keys}
    
    df = pd.DataFrame([val_row, test_row], index=["Validation", "Test"])
    print("\nFinal Performance Metrics:")
    print(df.to_markdown(floatfmt=".4f"))
    
    # Log to wandb
    wandb.log({"Final_Performance_Table": wandb.Table(dataframe=df.reset_index())})
    for k in metrics_keys:
        wandb.run.summary[f"test_best_{k}_{asp}"] = test_metrics[f"{k}_{asp}"]

    wandb.finish()
    return test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    parser.add_argument("--embedding", default="concat",
                        choices=["esm", "contvar", "concat"],
                        help="Embedding type: esm, contvar, or concat")
    args = parser.parse_args()
    test_eval(DecoderConfig(aspect=args.aspect, embedding_type=args.embedding))
