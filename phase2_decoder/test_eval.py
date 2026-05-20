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

    test_ds = GOAnnotationDataset(
        cfg.embeddings_h5, annotations, go_vocab,
        split="test",
        uniref_tsv=cfg.uniref_tsv,
        split_json=cfg.split_json,
    )
    print(f"Test: {len(test_ds)} proteins")

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    print(f"Loading checkpoint: {cfg.decoder_checkpoint}")
    model = FFNDecoder(cfg.encoder_output_dim, n_classes,
                       cfg.hidden_dims, cfg.dropout).to(DEVICE)
    model.load_state_dict(torch.load(cfg.decoder_checkpoint, map_location=DEVICE))
    model.eval()

    wandb.login(key=cfg.wandb_api_key)
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name="test-eval",
        config={
            "checkpoint": cfg.decoder_checkpoint,
            "threshold":  cfg.eval_threshold,
            "test_size":  len(test_ds),
        },
    )

    m = evaluate(model, test_loader, DEVICE, threshold=cfg.eval_threshold)

    print(
        f"\nTest | threshold={cfg.eval_threshold} | "
        f"mAP={m['mAP']:.4f} | F1={m['F1_macro']:.4f} | "
        f"Precision={m['Precision']:.4f} | Recall={m['Recall']:.4f} | "
        f"Accuracy={m['Accuracy']:.4f} | MCC={m['MCC']:.4f}"
    )

    wandb.log({
        "test/mAP":       m["mAP"],
        "test/F1_macro":  m["F1_macro"],
        "test/Precision": m["Precision"],
        "test/Recall":    m["Recall"],
        "test/Accuracy":  m["Accuracy"],
        "test/MCC":       m["MCC"],
    })
    for k, v in m.items():
        wandb.run.summary[f"test/{k}"] = v

    wandb.finish()
    return m


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    args = parser.parse_args()
    test_eval(DecoderConfig(aspect=args.aspect))
