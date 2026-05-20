"""
Decoder training.

Usage:
    python phase2_decoder/train.py
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from phase2_decoder.config import DecoderConfig
from phase2_decoder.data.parse_goa import parse_goa_tsv, build_go_vocab
from phase2_decoder.data.dataset import GOAnnotationDataset
from phase2_decoder.models.ffn_decoder import FFNDecoder
from phase2_decoder.evaluate import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Class weights ────────────────────────────────────────────────────────────

def compute_pos_weights(dataset: GOAnnotationDataset,
                        cfg: DecoderConfig) -> torch.Tensor:
    """
    For each GO term:
        pos_weight[i] = n_negative[i] / n_positive[i]

    Keep loss high for rare GO terms → force the model to learn them.
    An extra multiplier is applied to NULL_FUNCTION (very few examples).
    """
    labels = dataset.labels                  # [N, N_go] — already in RAM

    pos = labels.sum(dim=0)                  # [N_go]
    neg = len(labels) - pos

    pos_weight = neg / (pos + 1e-8)
    pos_weight[-1] *= cfg.null_function_weight   # extra weight for NULL_FUNCTION
    pos_weight = pos_weight.clamp(max=cfg.pos_weight_clamp)

    return pos_weight


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -float("inf")
        self.counter    = 0

    def step(self, score: float) -> bool:
        """Returns True → stop training."""
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Main training function ───────────────────────────────────────────────────

def train(cfg: DecoderConfig = None):
    if cfg is None:
        cfg = DecoderConfig()

    # wandb başlat
    wandb.login(key=cfg.wandb_api_key)
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config={
            "encoder_output_dim":  cfg.encoder_output_dim,
            "hidden_dims":         cfg.hidden_dims,
            "dropout":             cfg.dropout,
            "min_go_freq":         cfg.min_go_freq,
            "null_function_weight":cfg.null_function_weight,
            "pos_weight_clamp":    cfg.pos_weight_clamp,
            "lr":                  cfg.lr,
            "weight_decay":        cfg.weight_decay,
            "batch_size":          cfg.batch_size,
            "epochs":              cfg.epochs,
            "early_stop_patience": cfg.early_stop_patience,
        },
    )

    # 1. GO annotation parse
    print(f"Parsing GO annotations (aspect={cfg.aspect})...")
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect)

    # 2. Vocabulary
    print("Building vocabulary...")
    go_vocab = build_go_vocab(annotations, min_freq=cfg.min_go_freq,
                              save_path=cfg.go_vocab_json)
    n_classes = len(go_vocab)
    print(f"GO vocab size: {n_classes}")

    # 3. Datasets
    train_ds = GOAnnotationDataset(cfg.embeddings_h5, annotations, go_vocab,
                                   split="train",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json)
    val_ds   = GOAnnotationDataset(cfg.embeddings_h5, annotations, go_vocab,
                                   split="val",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # 4. Class weights
    print("Computing pos_weight...")
    pos_weight = compute_pos_weights(train_ds, cfg)

    # 5. Model, loss, optimizer, scheduler
    decoder   = FFNDecoder(cfg.encoder_output_dim, n_classes,
                           cfg.hidden_dims, cfg.dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(decoder.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=cfg.epochs)
    stopper   = EarlyStopper(patience=cfg.early_stop_patience)

    best_map = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        decoder.train()
        train_loss = 0.0

        for emb, label in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            emb   = emb.to(DEVICE)
            label = label.to(DEVICE)

            logits = decoder(emb)
            loss   = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────────
        metrics = evaluate(decoder, val_loader, DEVICE, criterion=criterion)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_mAP={metrics['mAP']:.4f} | "
            f"val_F1={metrics['F1_macro']:.4f} | "
            f"val_Acc={metrics['Accuracy']:.4f} | "
            f"val_Prec={metrics['Precision']:.4f} | "
            f"val_Rec={metrics['Recall']:.4f} | "
            f"val_MCC={metrics['MCC']:.4f} | "
            f"LR={current_lr:.2e}"
        )

        wandb.log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "val/loss":       metrics["val_loss"],
            "val/mAP":        metrics["mAP"],
            "val/F1_macro":   metrics["F1_macro"],
            "val/Accuracy":   metrics["Accuracy"],
            "val/Precision":  metrics["Precision"],
            "val/Recall":     metrics["Recall"],
            "val/MCC":        metrics["MCC"],
            "train/lr":       current_lr,
        })

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(decoder.state_dict(), cfg.decoder_checkpoint)
            print(f"  → saved (best mAP={best_map:.4f})")
            wandb.run.summary["best_mAP"]       = best_map
            wandb.run.summary["best_epoch"]     = epoch
            wandb.run.summary["best_F1_macro"]  = metrics["F1_macro"]
            wandb.run.summary["best_Accuracy"]  = metrics["Accuracy"]
            wandb.run.summary["best_Precision"] = metrics["Precision"]
            wandb.run.summary["best_Recall"]    = metrics["Recall"]
            wandb.run.summary["best_MCC"]       = metrics["MCC"]

        if stopper.step(metrics["mAP"]):
            print(f"Early stopping triggered (epoch={epoch})")
            break

    wandb.finish()
    return best_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    args = parser.parse_args()
    train(DecoderConfig(aspect=args.aspect))
