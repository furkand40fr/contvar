"""
Decoder evaluation metrics.

mAP        : Vectorised numpy implementation — much faster than sklearn's per-class loop.
F1_macro   : Count as positive if sigmoid(logit) > threshold, compute macro F1.
Accuracy   : Per-class (TP+TN) / N, macro average.
Precision  : Macro precision.
Recall     : Macro recall.
MCC        : Matthews Correlation Coefficient, macro average.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union


def _fast_map(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Vectorised mean Average Precision.
    labels, probs : [N, C]  — columns with no positives must be removed beforehand.
    """
    N = labels.shape[0]
    order         = np.argsort(-probs, axis=0)                      # [N, C]
    sorted_labels = np.take_along_axis(labels, order, axis=0)       # [N, C]
    cumpos        = np.cumsum(sorted_labels, axis=0)                 # [N, C]
    rank          = np.arange(1, N + 1, dtype=np.float32)[:, None]  # [N, 1]
    precision     = cumpos / rank                                    # [N, C]
    ap            = (precision * sorted_labels).sum(axis=0) / (sorted_labels.sum(axis=0) + 1e-8)
    return float(ap.mean())


def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, aspect: str,
             threshold: Union[float, str] = "sweep",
             criterion: torch.nn.Module = None,
             propagation_mat: np.ndarray = None) -> dict:
    """
    Parameters
    ----------
    propagation_mat : np.ndarray, optional, shape (N_go, N_go)
        Binary GO hierarchy propagation matrix built by
        GOHierarchy.build_propagation_matrix().  A[i, j] = 1 means term j
        is an ancestor of term i.  When provided, predicted probabilities
        are propagated upward through the DAG (max-pooling over
        descendants) before metric computation.
    """

    asp = aspect.lower()

    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0

    with torch.no_grad():
        for emb, label in loader:
            emb_d  = emb.to(device)
            logits = model(emb_d).cpu()
            all_logits.append(logits)
            all_labels.append(label)

            if criterion is not None:
                total_loss += criterion(logits.to(device), label.to(device)).item()
                n_batches  += 1

    logits = torch.cat(all_logits).numpy()   # [N, N_go]
    labels = torch.cat(all_labels).numpy()   # [N, N_go]
    probs  = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # ── GO hierarchy propagation (postprocessing) ────────────────────────
    if propagation_mat is not None:
        # propagation_mat: (N_go, N_go), A[i,j]=1 if j is ancestor of i
        # For each ancestor j, take max prob over all its descendants i
        # prop_probs[b, j] = max_{i : A[i,j]=1} probs[b, i]
        N_go = probs.shape[1]
        prop_probs = np.zeros_like(probs)
        for j in range(N_go):
            desc_mask = propagation_mat[:, j].astype(bool)  # rows where j is ancestor
            if desc_mask.any():
                prop_probs[:, j] = probs[:, desc_mask].max(axis=1)
            else:
                prop_probs[:, j] = probs[:, j]
        probs = prop_probs

    # Exclude GO terms with no positive examples from metric computation
    valid    = labels.sum(axis=0) > 0
    labels_v = labels[:, valid]
    probs_v  = probs[:, valid]
    N        = labels_v.shape[0]

    map_score = _fast_map(labels_v, probs_v)

    if threshold == "sweep":
        best_f1 = -1.0
        best_thr = 0.5
        for t in np.arange(0.05, 1.00, 0.05):
            preds_t = (probs_v > t).astype(np.float32)
            tp_t    = (preds_t * labels_v).sum(axis=0)
            fp_t    = (preds_t * (1 - labels_v)).sum(axis=0)
            fn_t    = ((1 - preds_t) * labels_v).sum(axis=0)
            f1_c    = 2 * tp_t / (2 * tp_t + fp_t + fn_t + 1e-8)
            f1_t    = float(f1_c.mean())
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thr = t
        threshold = best_thr

    preds = (probs_v > threshold).astype(np.float32)
    tp    = (preds * labels_v).sum(axis=0)
    tn    = ((1 - preds) * (1 - labels_v)).sum(axis=0)
    fp    = (preds * (1 - labels_v)).sum(axis=0)
    fn    = ((1 - preds) * labels_v).sum(axis=0)

    # F1
    f1_per_class = 2 * tp / (2 * tp + fp + fn + 1e-8)
    f1           = float(f1_per_class.mean())

    # Precision
    precision_per_class = tp / (tp + fp + 1e-8)
    precision           = float(precision_per_class.mean())

    # Recall
    recall_per_class = tp / (tp + fn + 1e-8)
    recall           = float(recall_per_class.mean())

    # Accuracy (per-class subset accuracy: (TP+TN)/N)
    accuracy_per_class = (tp + tn) / (N + 1e-8)
    accuracy           = float(accuracy_per_class.mean())

    # MCC (Matthews Correlation Coefficient)
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc_per_class = mcc_num / mcc_den
    mcc           = float(mcc_per_class.mean())

    results = {
        f"mAP_{asp}":       map_score,
        f"F1_macro_{asp}":  f1,
        f"Precision_{asp}": precision,
        f"Recall_{asp}":    recall,
        f"Accuracy_{asp}":  accuracy,
        f"MCC_{asp}":       mcc,
        f"threshold_{asp}": threshold,
    }

    if criterion is not None:
        results["val_loss"] = total_loss / (n_batches + 1e-8)

    return results
