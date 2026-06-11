import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from contvar.config import ProjectConfig, ensure_dms_triplets_unzipped
from contvar.data.mapper import TripletDataPathMapper, DmsProteinSplitError
from contvar.data.dataset import TripletProteinGraphDataset, ExhaustiveTripletDataset
from contvar.data.collate import triplet_collate
from contvar.model import DeepProteinGAT
from contvar.losses import StandardTripletLoss, SemiHardMiningTripletLoss
from contvar.mining import streaming_mining_batch_iterator
from contvar.metrics import compute_detailed_metrics, compute_embedding_stats
from contvar.utils import load_all_embeddings
from contvar.go_pretraining import run_go_pretraining


def _load_model_checkpoint(model, checkpoint_path, device, strict=True):
    """Load either a raw state_dict or a checkpoint dict into the model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("mutation_attention_")
    }
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(
                "[Checkpoint] Missing keys initialized from current model: "
                + ", ".join(missing)
            )
        if unexpected:
            print(
                "[Checkpoint] Unexpected keys ignored from checkpoint: "
                + ", ".join(unexpected)
            )


def _ensure_parent_dir(path):
    """Create the parent directory for a file path when needed."""
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _should_save_stage2_epoch_checkpoint(epoch_number, extra_epochs, interval):
    """Return whether Stage-2 should persist a snapshot for this epoch."""
    if epoch_number in extra_epochs:
        return True
    return interval > 0 and epoch_number % interval == 0


def _format_stage2_epoch_checkpoint_path(template, epoch_number):
    """Build a Stage-2 epoch checkpoint path from a format template."""
    if not template:
        return None
    return template.format(epoch=epoch_number)


def _build_lr_scheduler(optimizer, cfg):
    """Build the Stage-2 learning-rate scheduler."""
    scheduler_type = str(getattr(cfg, "lr_scheduler", "warmup_cosine")).lower()
    decay = float(getattr(cfg, "lr_decay", 1.0))
    min_lr = float(getattr(cfg, "min_lr", 0.0))
    base_lr = float(getattr(cfg, "lr", 0.0))
    total_epochs = int(getattr(cfg, "epochs", 0))
    warmup_epochs = int(getattr(cfg, "lr_warmup_epochs", 0))
    warmup_start_factor = float(getattr(cfg, "lr_warmup_start_factor", 0.1))

    if scheduler_type in ("none", "off", "disabled"):
        return None
    if base_lr <= 0:
        return None
    if decay <= 0:
        raise ValueError(f"lr_decay must be positive, got {decay}")
    if min_lr < 0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}")
    if min_lr > base_lr:
        raise ValueError(f"min_lr ({min_lr}) cannot be greater than lr ({base_lr})")
    if warmup_epochs < 0:
        raise ValueError(f"lr_warmup_epochs must be non-negative, got {warmup_epochs}")
    if not 0 < warmup_start_factor <= 1:
        raise ValueError(
            "lr_warmup_start_factor must be in the interval (0, 1], "
            f"got {warmup_start_factor}"
        )

    min_factor = min_lr / base_lr if min_lr > 0 else 0.0

    if scheduler_type == "exponential":
        if decay >= 1.0:
            return None

        def lr_lambda(epoch):
            return max(decay ** epoch, min_factor)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_type not in ("cosine", "warmup_cosine"):
        raise ValueError(
            "lr_scheduler must be one of: none, exponential, cosine, warmup_cosine "
            f"(got {scheduler_type!r})"
        )

    if scheduler_type == "cosine":
        warmup_epochs = 0

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = epoch / max(1, warmup_epochs - 1)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)

        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine_factor

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(model, loader, criterion, device, margin=0.3):
    """Evaluate model on a given dataloader with both global and local loss"""
    model.eval()
    total_loss = 0
    total_loss_g = 0
    total_loss_local = 0
    valid_batches = 0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives = batch
            ba = ba.to(device)
            bp = bp.to(device)
            bn = bn.to(device)
            neg_counts = neg_counts.to(device)
            mut_pos_positive = mut_pos_positive.to(device)
            mut_pos_negatives = mut_pos_negatives.to(device)

            # Forward anchor ONCE, reuse node features for local extraction
            ea_g, ea_l_pos, anchor_ctx = model.forward_with_nodes(ba, mut_pos=mut_pos_positive)
            ep_g, ep_l = model(bp, mut_pos=mut_pos_positive)
            en_g, en_l = model(bn, mut_pos=mut_pos_negatives)

            # Global loss
            loss_g, neg_dist, en_neg, mining_stats = criterion(ea_g, ep_g, en_g, neg_counts)

            # Local loss
            hardest_indices = mining_stats["hardest_indices"]
            cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
            flat_idx = cumsum + hardest_indices
            mut_pos_neg_selected = mut_pos_negatives[flat_idx]

            la_at_pos = ea_l_pos
            a_nodes, a_batch, a_resnum = anchor_ctx
            la_at_neg = model._extract_local(a_nodes, a_batch, a_resnum, mut_pos_neg_selected)
            zn_l_selected = en_l[flat_idx]

            B = la_at_pos.size(0)
            z_wt_l = torch.cat([la_at_pos, la_at_neg], dim=0)
            z_mut_l = torch.cat([ep_l, zn_l_selected], dim=0)
            lbl = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B, device=device)
            ])

            d_local = F.pairwise_distance(z_wt_l, z_mut_l, p=2)
            loss_attract = lbl * (d_local ** 2)
            loss_repel = (1.0 - lbl) * (F.relu(margin - d_local) ** 2)
            loss_l = (loss_attract + loss_repel).mean()

            # Combined loss
            loss = (loss_g + loss_l) / 2

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_local += loss_l.item()
            valid_batches += 1

            batch_metrics = compute_detailed_metrics(ea_g, ep_g, en_neg)
            batch_metrics["loss"] = loss.item()
            batch_metrics["loss_global"] = loss_g.item()
            batch_metrics["loss_local"] = loss_l.item()
            all_metrics.append(batch_metrics)

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0

    aggregated = {"loss": avg_loss}
    if all_metrics:
        metric_keys = [k for k in all_metrics[0].keys() if k != "loss"]
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = np.mean(values) if values else 0.0

    return aggregated


def train_pipeline(config=None, force=False, data_root=None,
                   embeddings_path=None, device=None, data_zip=None):
    """
    Main training pipeline.

    Args:
        config: dict of config overrides (e.g. from wandb sweep)
        force: If True, reprocess all protein graphs from scratch
        data_root: Path to protein_triplets_data directory
        embeddings_path: Path to ESM2 embeddings h5 file
        device: torch device (auto-detected if None)
        data_zip: Optional path to protein_triplets zip (Colab); used if folder missing
    """
    # Initialize config
    cfg = ProjectConfig()

    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    run = wandb.init(
        project="ContVAR-Project",
        config=vars(cfg),
        reinit=True,
        settings=wandb.Settings(_disable_stats=True)
    )

    # Resolve paths
    if data_root is None:
        from contvar.config import setup_environment
        env = setup_environment()
        data_root = env['data_root']
        if embeddings_path is None:
            embeddings_path = env['embeddings_path']
        if data_zip is None:
            data_zip = env.get("data_zip")

    print(
        f"Training with LR: {cfg.lr}, Scheduler: {cfg.lr_scheduler}, "
        f"Warmup epochs: {cfg.lr_warmup_epochs}, Min LR: {cfg.min_lr}, "
        f"Hidden: {cfg.hidden_dim}, Heads: {cfg.heads}"
    )
    print(f"Streaming Mining: chunk_size={cfg.mining_chunk_size}, max_negatives={cfg.max_negatives}")
    print(f"Gradient Accumulation: {cfg.grad_accumulation_steps} steps (effective batch = {cfg.mining_batch_size * cfg.grad_accumulation_steps})")
    print(f"Eval Batch Size: {cfg.eval_batch_size}")
    print(f"Local Loss: Contrastive (attract good / repel bad at mutation position)")
    print("Global Pooling: global_mean_pool")
    print(f"DMS protein split: {cfg.dms_protein_split_json_path}")
    print(f"Stage-2 best checkpoint: {cfg.stage2_best_model_path}")
    print(f"Stage-2 last checkpoint: {cfg.stage2_last_model_path}")
    if getattr(cfg, "stage2_epoch_checkpoint_template", None):
        print(
            "Stage-2 epoch checkpoint template: "
            f"{cfg.stage2_epoch_checkpoint_template}"
        )

    shared_embeddings = None
    if force and embeddings_path:
        shared_embeddings = load_all_embeddings(embeddings_path)

    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    model = DeepProteinGAT(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        heads=cfg.heads,
        edge_dim=cfg.edge_attr_dim,
    ).to(device)

    init_checkpoint_path = getattr(cfg, "go_phase0_init_checkpoint_path", None)
    if init_checkpoint_path:
        if not os.path.isfile(init_checkpoint_path):
            raise FileNotFoundError(
                f"GO phase-0 init checkpoint not found: {init_checkpoint_path}"
            )
        _load_model_checkpoint(model, init_checkpoint_path, device, strict=False)
        print(f"[Phase0] Initialized model from checkpoint: {init_checkpoint_path}")
        run.summary["phase0_init_checkpoint_path"] = init_checkpoint_path

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = _build_lr_scheduler(optimizer, cfg)

    standard_criterion = StandardTripletLoss(margin=cfg.margin)
    semihard_criterion = SemiHardMiningTripletLoss(margin=cfg.margin)

    val_criterion = standard_criterion

    # Phase 0: GO semantic pretraining (Drive TSVs + prebuilt PyG graphs). No DMS zip yet.
    if getattr(cfg, "go_phase0_epochs", 0) > 0:
        run_go_pretraining(model, cfg, device)

    # DMS triplets (zip on Colab if needed), then stage-2 training on fixed splits.
    print("\n=== DMS data: unzip if needed, then load fixed protein split ===")
    ensure_dms_triplets_unzipped(data_root, data_zip)
    try:
        mapper = TripletDataPathMapper(
            data_root,
            split_json_path=getattr(cfg, "dms_protein_split_json_path", None),
        )
    except DmsProteinSplitError as exc:
        print(f"\nDMS split configuration error: {exc}")
        wandb.finish()
        return None, None, None

    if not mapper.triplets:
        print("No data found!")
        wandb.finish()
        return None, None, None

    # =========================================================================
    # CREATE DATASETS
    # =========================================================================
    print("\n=== Stage-2 Setup: Streaming Train + Exhaustive Val/Test ===")

    main_train_dataset = TripletProteinGraphDataset(
        mapper, root=data_root, config=cfg, split='train',
        esm2_embedding_path=embeddings_path,
        force=force, preloaded_embeddings=shared_embeddings
    )

    val_dataset = ExhaustiveTripletDataset(
        mapper, root=data_root, config=cfg, split='val',
        preloaded_embeddings=shared_embeddings
    )
    test_dataset = ExhaustiveTripletDataset(
        mapper, root=data_root, config=cfg, split='test',
        preloaded_embeddings=shared_embeddings
    )

    # =========================================================================
    # CREATE DATALOADERS
    # =========================================================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=triplet_collate,
        num_workers=getattr(cfg, "num_workers", 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=triplet_collate,
        num_workers=getattr(cfg, "num_workers", 0)
    )

    processed_dir = os.path.join(data_root, "processed")
    num_train_batches = (
        len(main_train_dataset.triplets) + cfg.mining_batch_size - 1
    ) // cfg.mining_batch_size

    print(f"\nDataset sizes:")
    print(f"  Train families: {len(main_train_dataset.triplets):,}")
    print(f"  Val triplets (exhaustive): {len(val_dataset):,}")
    print(f"  Test triplets (exhaustive): {len(test_dataset):,}")

    print("\n" + "="*60)
    print("STARTING STAGE-2 DMS TRAINING")
    print("="*60)

    best_val_loss = float('inf')
    best_epoch = None
    best_model_path = (
        getattr(cfg, "stage2_best_model_path", None) or "model_best_loss.pt"
    )
    last_model_path = (
        getattr(cfg, "stage2_last_model_path", None) or "model_last.pt"
    )
    epoch_checkpoint_template = getattr(
        cfg, "stage2_epoch_checkpoint_template", None
    )
    epoch_checkpoint_extra_epochs = {
        int(epoch)
        for epoch in (
            getattr(cfg, "stage2_epoch_checkpoint_extra_epochs", (80,)) or ()
        )
    }
    epoch_checkpoint_interval = int(
        getattr(cfg, "stage2_epoch_checkpoint_interval", 100) or 0
    )
    _ensure_parent_dir(best_model_path)
    _ensure_parent_dir(last_model_path)

    for epoch in range(cfg.epochs):
        epoch_number = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]
        train_loader = streaming_mining_batch_iterator(
            model, main_train_dataset.triplets, processed_dir, device, cfg
        )
        criterion = semihard_criterion
        current_batch_size = cfg.mining_batch_size

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg.epochs} | Stage-2 Streaming Semi-Hard Mining")
        print(
            f"Batch size: {current_batch_size} | Train batches: ~{num_train_batches} | "
            f"LR: {current_lr:.6g}"
        )
        print(f"{'='*60}")

        # =====================================================================
        # TRAINING
        # =====================================================================
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_loss_g = 0
        total_loss_local = 0
        valid_batches = 0
        train_metrics_list = []

        epoch_streaming_hard_total = 0
        epoch_streaming_semi_hard_total = 0
        epoch_streaming_evaluated_total = 0
        epoch_streaming_qualifying_total = 0

        epoch_local_easy_total = 0
        epoch_local_semi_hard_total = 0
        epoch_local_hard_total = 0

        accum_steps = cfg.grad_accumulation_steps

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, total=num_train_batches)

        for batch in pbar:
            if batch is None:
                continue

            ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives, streaming_info = batch

            if ba.num_graphs < 2:
                continue

            ba = ba.to(device)
            bp = bp.to(device)
            bn = bn.to(device)
            neg_counts = neg_counts.to(device)
            mut_pos_positive = mut_pos_positive.to(device)
            mut_pos_negatives = mut_pos_negatives.to(device)

            # Forward anchor ONCE, reuse node features for local extraction
            ea_g, ea_l_pos, anchor_ctx = model.forward_with_nodes(ba, mut_pos=mut_pos_positive)
            ep_g, ep_l = model(bp, mut_pos=mut_pos_positive)
            en_g, en_l = model(bn, mut_pos=mut_pos_negatives)

            loss_g, neg_dist, en_neg, mining_stats = criterion(ea_g, ep_g, en_g, neg_counts)
            hardest_indices = mining_stats["hardest_indices"]
            cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
            flat_idx = cumsum + hardest_indices
            mut_pos_neg_selected = mut_pos_negatives[flat_idx]

            # =================================================================
            # LOCAL CONTRASTIVE LOSS
            # =================================================================
            la_at_pos = ea_l_pos  # already extracted above
            a_nodes, a_batch, a_resnum = anchor_ctx
            la_at_neg = model._extract_local(a_nodes, a_batch, a_resnum, mut_pos_neg_selected)
            zn_l_selected = en_l[flat_idx]

            B = la_at_pos.size(0)

            z_wt_l = torch.cat([la_at_pos, la_at_neg], dim=0)
            z_mut_l = torch.cat([ep_l, zn_l_selected], dim=0)
            lbl = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B, device=device)
            ])

            d_local = F.pairwise_distance(z_wt_l, z_mut_l, p=2)

            loss_attract = lbl * (d_local ** 2)
            loss_repel = (1.0 - lbl) * (F.relu(cfg.margin - d_local) ** 2)
            loss_l = (loss_attract + loss_repel).mean()

            d_pos_l = d_local[:B]
            d_neg_l = d_local[B:]

            # Local contrastive mining stats
            with torch.no_grad():
                neg_mask_l = (lbl == 0)
                if neg_mask_l.sum() > 0:
                    d_neg_stats = d_local[neg_mask_l]
                    batch_local_easy = int((d_neg_stats > cfg.margin).sum().item())
                    batch_local_hard = int((d_neg_stats < cfg.margin * 0.5).sum().item())
                    batch_local_semi = int(((d_neg_stats >= cfg.margin * 0.5) & (d_neg_stats <= cfg.margin)).sum().item())
                else:
                    batch_local_easy = batch_local_hard = batch_local_semi = 0

                epoch_local_easy_total += batch_local_easy
                epoch_local_hard_total += batch_local_hard
                epoch_local_semi_hard_total += batch_local_semi

            # =================================================================
            # COMBINED LOSS
            # =================================================================
            loss = (loss_g + loss_l) / 2
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if valid_batches % accum_steps == (accum_steps - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                batch_metrics = compute_detailed_metrics(ea_g, ep_g, en_neg)
                dist_pos = F.pairwise_distance(ea_g, ep_g)
                train_metrics_list.append(batch_metrics)

            batch_log = {
                "train/batch_loss": loss.item(),
                "train/batch_loss_global": loss_g.item(),
                "train/batch_loss_local": loss_l.item(),
                "train/Alignment_batch": batch_metrics["Alignment"],
                "train/MRR_batch": batch_metrics["MRR"],
                "train/Uniformity_batch": batch_metrics["Uniformity"],
                "train/avg_pos_dist": dist_pos.mean().item(),
                "train/avg_neg_dist": neg_dist.mean().item(),
                "train/dist_margin": (neg_dist.mean() - dist_pos.mean()).item(),
                "train/avg_pos_dist_local": d_pos_l.mean().item(),
                "train/avg_neg_dist_local": d_neg_l.mean().item(),
                "train/dist_margin_local": (d_neg_l.mean() - d_pos_l.mean()).item(),
                
                "local/batch_easy": batch_local_easy,
                "local/batch_semi_hard": batch_local_semi,
                "local/batch_hard": batch_local_hard,
            }

            if streaming_info is not None:
                batch_log["mining/batch_hard_count"] = streaming_info["streaming_hard"]
                batch_log["mining/batch_semi_hard_count"] = streaming_info["streaming_semi_hard"]
                batch_log["mining/batch_total_evaluated"] = streaming_info["total_evaluated"]
                batch_log["mining/batch_total_qualifying"] = streaming_info["total_qualifying"]
                batch_log["mining/batch_qualifying_ratio"] = (
                    streaming_info["total_qualifying"] / streaming_info["total_evaluated"]
                    if streaming_info["total_evaluated"] > 0 else 0
                )

            wandb.log(batch_log)

            if streaming_info is not None:
                epoch_streaming_hard_total += streaming_info["streaming_hard"]
                epoch_streaming_semi_hard_total += streaming_info["streaming_semi_hard"]
                epoch_streaming_evaluated_total += streaming_info["total_evaluated"]
                epoch_streaming_qualifying_total += streaming_info["total_qualifying"]

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_local += loss_l.item()
            valid_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mrr': f'{batch_metrics["MRR"]:.3f}',
            })

        # Flush remaining accumulated gradients at end of epoch
        if valid_batches % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / valid_batches if valid_batches > 0 else 0
        avg_train_loss_g = total_loss_g / valid_batches if valid_batches > 0 else 0
        avg_train_loss_local = total_loss_local / valid_batches if valid_batches > 0 else 0
        epoch_duration_sec = time.time() - epoch_start

        train_epoch_metrics = {}
        if train_metrics_list:
            for key in train_metrics_list[0].keys():
                values = [m[key] for m in train_metrics_list]
                train_epoch_metrics[key] = np.mean(values)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        val_metrics = evaluate(model, val_loader, val_criterion, device, margin=cfg.margin)

        embedding_stats = compute_embedding_stats(
            model, val_loader, device, val_criterion, max_batches=20
        )
        epoch_log = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_loss_global": avg_train_loss_g,
            "train/epoch_loss_local": avg_train_loss_local,
            "train/epoch_Alignment": train_epoch_metrics.get("Alignment", 0),
            "train/epoch_MRR": train_epoch_metrics.get("MRR", 0),
            "train/epoch_Uniformity": train_epoch_metrics.get("Uniformity", 0),
            "train/epoch_duration_sec": epoch_duration_sec,
            "train/lr": current_lr,

            "val/loss": val_metrics.get("loss", 0),
            "val/loss_global": val_metrics.get("loss_global", 0),
            "val/loss_local": val_metrics.get("loss_local", 0),
            "val/MRR": val_metrics.get("MRR", 0),
            "val/Alignment": val_metrics.get("Alignment", 0),
            "val/Uniformity": val_metrics.get("Uniformity", 0),

        }

        for k, v in embedding_stats.items():
            epoch_log[f"embedding_stats/{k}"] = v

        epoch_log["mining/epoch_hard_total"] = epoch_streaming_hard_total
        epoch_log["mining/epoch_semi_hard_total"] = epoch_streaming_semi_hard_total
        epoch_log["mining/epoch_total_evaluated"] = epoch_streaming_evaluated_total
        epoch_log["mining/epoch_total_qualifying"] = epoch_streaming_qualifying_total
        epoch_log["mining/epoch_qualifying_ratio"] = (
            epoch_streaming_qualifying_total / epoch_streaming_evaluated_total
            if epoch_streaming_evaluated_total > 0 else 0
        )
        epoch_log["mining/epoch_easy_total"] = (
            epoch_streaming_evaluated_total - epoch_streaming_qualifying_total
        )

        epoch_log["local/epoch_easy_total"] = epoch_local_easy_total
        epoch_log["local/epoch_semi_hard_total"] = epoch_local_semi_hard_total
        epoch_log["local/epoch_hard_total"] = epoch_local_hard_total

        # Save best model
        if val_metrics.get("loss", float('inf')) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch_number
            model_name = best_model_path
            torch.save(model.state_dict(), model_name)

            artifact = wandb.Artifact(
                name=f"ContVAR-Best-Model-{wandb.run.id}",
                type="model",
                description=f"Best model at epoch {epoch_number} with val_loss {best_val_loss:.4f}"
            )
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)
            epoch_log["best_model_saved"] = True
        epoch_log["val/best_loss_so_far"] = best_val_loss
        epoch_log["val/best_epoch_so_far"] = best_epoch if best_epoch is not None else 0

        epoch_checkpoint_path = None
        if _should_save_stage2_epoch_checkpoint(
            epoch_number,
            epoch_checkpoint_extra_epochs,
            epoch_checkpoint_interval,
        ):
            epoch_checkpoint_path = _format_stage2_epoch_checkpoint_path(
                epoch_checkpoint_template, epoch_number
            )
            if epoch_checkpoint_path:
                _ensure_parent_dir(epoch_checkpoint_path)
                torch.save(model.state_dict(), epoch_checkpoint_path)
                epoch_log["checkpoint/epoch_snapshot_saved"] = True
                epoch_log["checkpoint/epoch_snapshot_epoch"] = epoch_number
                print(
                    "[Stage2] Saved epoch checkpoint "
                    f"to {epoch_checkpoint_path}"
                )

        wandb.log(epoch_log)
        if lr_scheduler is not None:
            lr_scheduler.step()

        saved_str = "(Saved)" if epoch_log.get("best_model_saved") else ""
        epoch_saved_str = "(Epoch checkpoint saved)" if epoch_checkpoint_path else ""
        print(f"[Stage2] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
              f"Val MRR: {val_metrics.get('MRR', 0):.4f} | "
              f"Local[E:{epoch_local_easy_total} S:{epoch_local_semi_hard_total} H:{epoch_local_hard_total}] "
              f"{saved_str} {epoch_saved_str}")

    # Save last epoch model
    torch.save(model.state_dict(), last_model_path)
    print(f"Saved last epoch model to {last_model_path}")

    if best_epoch is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_val_metrics = evaluate(model, val_loader, val_criterion, device, margin=cfg.margin)
        best_test_metrics = evaluate(model, test_loader, val_criterion, device, margin=cfg.margin)

        final_log = {
            "best/epoch": best_epoch,
            "best/val_loss": best_val_metrics.get("loss", 0),
            "best/val_loss_global": best_val_metrics.get("loss_global", 0),
            "best/val_loss_local": best_val_metrics.get("loss_local", 0),
            "best/val_MRR": best_val_metrics.get("MRR", 0),
            "best/val_Alignment": best_val_metrics.get("Alignment", 0),
            "best/val_Uniformity": best_val_metrics.get("Uniformity", 0),
            "best/test_loss": best_test_metrics.get("loss", 0),
            "best/test_loss_global": best_test_metrics.get("loss_global", 0),
            "best/test_loss_local": best_test_metrics.get("loss_local", 0),
            "best/test_MRR": best_test_metrics.get("MRR", 0),
            "best/test_Alignment": best_test_metrics.get("Alignment", 0),
            "best/test_Uniformity": best_test_metrics.get("Uniformity", 0),
        }
        wandb.log(final_log)
        for key, value in final_log.items():
            run.summary[key] = value

        print(f"[Stage2] Best checkpoint (epoch {best_epoch}) | "
              f"Val Loss: {best_val_metrics.get('loss', 0):.4f} | "
              f"Test Loss: {best_test_metrics.get('loss', 0):.4f} | "
              f"Val MRR: {best_val_metrics.get('MRR', 0):.4f} | "
              f"Test MRR: {best_test_metrics.get('MRR', 0):.4f}")
    else:
        print("Warning: No best validation checkpoint was saved, so final test evaluation was skipped.")

    wandb.finish()
    print("\nTraining completed!")
    return model, mapper, processed_dir
