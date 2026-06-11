"""
Decoder training.

Usage:
    python phase2_decoder/train.py
"""

import json
import os
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
    train_ds = GOAnnotationDataset(cfg.esm_h5, cfg.contvar_h5, annotations, go_vocab,
                                   split="train",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   embedding_type=cfg.embedding_type,
                                   contvar_full_h5_path=cfg.contvar_full_h5)
    val_ds   = GOAnnotationDataset(cfg.esm_h5, cfg.contvar_h5, annotations, go_vocab,
                                   split="val",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   embedding_type=cfg.embedding_type,
                                   contvar_full_h5_path=cfg.contvar_full_h5)
    test_ds  = GOAnnotationDataset(cfg.esm_h5, cfg.contvar_h5, annotations, go_vocab,
                                   split="test",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   embedding_type=cfg.embedding_type,
                                   contvar_full_h5_path=cfg.contvar_full_h5)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
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

    asp         = cfg.aspect.lower()
    best_map    = 0.0
    best_epoch  = 0
    global_step = 0
    start_epoch = 1

    # ── GO hierarchy propagation matrix (built once, reused every epoch) ─────
    prop_mat = None
    if cfg.use_go_propagation:
        from phase2_decoder.data.go_hierarchy import GOHierarchy
        obo_path = cfg.obo_path
        if os.path.exists(obo_path):
            print(f"Building GO hierarchy propagation matrix from {obo_path}...")
            go_hier = GOHierarchy(obo_path)
            prop_mat = go_hier.build_propagation_matrix(go_vocab).numpy()
            print(f"  → Propagation matrix shape: {prop_mat.shape}, non-zero entries: {int(prop_mat.sum())}")
        else:
            print(f"  ⚠ go.obo not found at {obo_path}, skipping hierarchy propagation.")

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if getattr(cfg, '_resume', False) and cfg.decoder_checkpoint:
        import os
        ckpt_path = cfg.decoder_checkpoint
        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                # Full checkpoint
                decoder.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_map    = ckpt["best_map"]
                global_step = ckpt["global_step"]
                stopper.best    = ckpt.get("stopper_best", best_map)
                stopper.counter = ckpt.get("stopper_counter", 0)
                print(f"  → Resumed from epoch {ckpt['epoch']} (best mAP={best_map:.4f}, global_step={global_step})")
            else:
                decoder.load_state_dict(ckpt)
                print(f"  → Loaded model weights (old-style checkpoint, starting from epoch 1)")
        else:
            print(f"  ⚠ Checkpoint not found: {ckpt_path}, training from scratch.")

    for epoch in range(start_epoch, cfg.epochs + 1):
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

            train_loss  += loss.item()
            global_step += 1

            wandb.log({"train/step_loss": loss.item()}, step=global_step)

        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────────
        metrics = evaluate(decoder, val_loader, DEVICE, aspect=cfg.aspect,
                           criterion=criterion, propagation_mat=prop_mat)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"mAP_{asp}={metrics[f'mAP_{asp}']:.4f} | "
            f"F1_{asp}={metrics[f'F1_macro_{asp}']:.4f} | "
            f"Acc_{asp}={metrics[f'Accuracy_{asp}']:.4f} | "
            f"Prec_{asp}={metrics[f'Precision_{asp}']:.4f} | "
            f"Rec_{asp}={metrics[f'Recall_{asp}']:.4f} | "
            f"MCC_{asp}={metrics[f'MCC_{asp}']:.4f} | "
            f"thr_{asp}={metrics[f'threshold_{asp}']:.2f} | "
            f"LR={current_lr:.2e}"
        )

        wandb.log({
            "epoch":                   epoch,
            "train/loss":              train_loss,
            "val/loss":                metrics["val_loss"],
            f"val/mAP_{asp}":          metrics[f"mAP_{asp}"],
            f"val/F1_macro_{asp}":     metrics[f"F1_macro_{asp}"],
            f"val/Accuracy_{asp}":     metrics[f"Accuracy_{asp}"],
            f"val/Precision_{asp}":    metrics[f"Precision_{asp}"],
            f"val/Recall_{asp}":       metrics[f"Recall_{asp}"],
            f"val/MCC_{asp}":          metrics[f"MCC_{asp}"],
            f"val/threshold_{asp}":    metrics[f"threshold_{asp}"],
            "train/lr":                current_lr,
        }, step=global_step)

        if metrics[f"mAP_{asp}"] > best_map:
            best_map   = metrics[f"mAP_{asp}"]
            best_epoch = epoch
            torch.save({
                "model_state_dict":     decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":               epoch,
                "best_map":            best_map,
                "global_step":         global_step,
                "stopper_best":        stopper.best,
                "stopper_counter":     stopper.counter,
            }, cfg.decoder_checkpoint)
            print(f"  → saved (best mAP_{asp}={best_map:.4f})")
            wandb.run.summary[f"best_mAP_{asp}"]       = best_map
            wandb.run.summary["best_epoch"]             = epoch
            wandb.run.summary[f"best_F1_macro_{asp}"]   = metrics[f"F1_macro_{asp}"]
            wandb.run.summary[f"best_Accuracy_{asp}"]   = metrics[f"Accuracy_{asp}"]
            wandb.run.summary[f"best_Precision_{asp}"]  = metrics[f"Precision_{asp}"]
            wandb.run.summary[f"best_Recall_{asp}"]     = metrics[f"Recall_{asp}"]
            wandb.run.summary[f"best_MCC_{asp}"]        = metrics[f"MCC_{asp}"]

        if stopper.step(metrics[f"mAP_{asp}"]):
            print(f"Early stopping triggered (epoch={epoch})")
            break

    # ── Final Evaluation & Tabulation ───────────────────────────────────────────
    print("\n--- Training Completed ---")
    print(f"Loading best model from {cfg.decoder_checkpoint} for final evaluation...")
    ckpt = torch.load(cfg.decoder_checkpoint, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        decoder.load_state_dict(ckpt["model_state_dict"])
    else:
        decoder.load_state_dict(ckpt)


    val_metrics = evaluate(decoder, val_loader, DEVICE, aspect=cfg.aspect,
                           threshold=cfg.eval_threshold, propagation_mat=prop_mat)
    test_metrics = evaluate(decoder, test_loader, DEVICE, aspect=cfg.aspect,
                            threshold=cfg.eval_threshold, propagation_mat=prop_mat)
    
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

    # Return rich dict for grid search, or just best_map for standalone use
    if getattr(cfg, '_return_metrics', False):
        return {
            "aspect": cfg.aspect,
            "embedding_type": cfg.embedding_type,
            "best_epoch": best_epoch,
            "best_val_map": best_map,
            "checkpoint": cfg.decoder_checkpoint,
            "hyperparams": {
                "lr": cfg.lr,
                "dropout": cfg.dropout,
                "hidden_dims": cfg.hidden_dims,
                "weight_decay": cfg.weight_decay,
                "pos_weight_clamp": cfg.pos_weight_clamp,
                "epochs": cfg.epochs,
                "early_stop_patience": cfg.early_stop_patience,
            },
            "val": {
                "mAP": val_metrics[f"mAP_{asp}"],
                "F1_macro": val_metrics[f"F1_macro_{asp}"],
                "MCC": val_metrics[f"MCC_{asp}"],
                "threshold": val_metrics[f"threshold_{asp}"],
            },
            "test": {
                "mAP": test_metrics[f"mAP_{asp}"],
                "F1_macro": test_metrics[f"F1_macro_{asp}"],
                "MCC": test_metrics[f"MCC_{asp}"],
                "threshold": test_metrics[f"threshold_{asp}"],
            },
        }
    return best_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    parser.add_argument("--embedding", default="concat",
                        choices=["esm", "contvar", "contvar_full", "concat", "concat_full"],
                        help="Embedding type: esm, contvar, contvar_full, concat, or concat_full")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last saved checkpoint")
    parser.add_argument("--propagation", action="store_true",
                        help="Enable GO hierarchy propagation as postprocessing")
    args = parser.parse_args()
    cfg = DecoderConfig(aspect=args.aspect, embedding_type=args.embedding)
    cfg._resume = args.resume
    cfg.use_go_propagation = args.propagation
    train(cfg)
