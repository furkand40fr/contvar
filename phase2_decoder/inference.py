"""
Variant function change prediction.

Since the WT protein's GO annotations are known from the database (GOA),
the decoder is run only on the variant embedding.

    wt_label  = 1.0  (annotated in GO database)  or  0.0  (absent)
    var_score = sigmoid(decoder(var_emb))  per GO term [0,1]
    ratio     = var_score / wt_label   (meaningful only for annotated GO terms)

    wt_label=1 and var_score < (1 - threshold)  →  LOSS   (function loss)
    wt_label=0 and var_score > threshold         →  GAIN   (novel function gain)
    otherwise                                    →  NEUTRAL

ContVAR_fitness (single scalar):
    Computed only over GO terms annotated in the WT.
    mean(var_score[wt_annotated]) / 1.0
    1.0 = all WT functions retained | 0.0 = complete function loss
"""

import json
import h5py
import torch
from phase2_decoder.models.ffn_decoder import FFNDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_decoder(checkpoint: str, go_vocab_json: str,
                 encoder_output_dim: int = 256,
                 hidden_dims: list[int] = None,
                 dropout: float = 0.3) -> tuple:
    """Loads the saved decoder and vocabulary."""
    with open(go_vocab_json) as f:
        go_vocab = json.load(f)

    n_classes = len(go_vocab)
    decoder   = FFNDecoder(encoder_output_dim, n_classes, hidden_dims, dropout)
    decoder.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    decoder.eval().to(DEVICE)

    return decoder, go_vocab


def load_embedding(h5_path: str, protein_id: str) -> torch.Tensor:
    """Loads a single protein embedding from an H5 file → [D]"""
    with h5py.File(h5_path, "r") as h5f:
        return torch.FloatTensor(h5f[protein_id][:])


def predict_function_change(wt_go_terms: set, var_emb: torch.Tensor,
                            decoder: torch.nn.Module, go_vocab: dict,
                            threshold: float = 0.1) -> dict:
    """
    wt_go_terms : set of str  — known GO terms of the WT protein (from database)
                                e.g. {"GO:0003700", "GO:0005488"}
    var_emb     : [D]         — variant encoder embedding

    Returns:
    {
        "GO:0003700": {"wt": 1, "var": 0.12, "status": "LOSS"},
        "GO:0009999": {"wt": 0, "var": 0.73, "status": "GAIN"},
        ...
        "ContVAR_fitness": 0.81   # computed over WT-annotated GO terms
    }
    """
    decoder.eval()
    with torch.no_grad():
        var_probs = torch.sigmoid(decoder(var_emb.unsqueeze(0).to(DEVICE))).squeeze().cpu()

    results = {}
    annotated_var_scores = []

    for go_term, idx in go_vocab.items():
        if go_term == "NULL_FUNCTION":
            continue

        wt_label = 1 if go_term in wt_go_terms else 0
        var_s    = var_probs[idx].item()

        if wt_label == 1 and var_s < (1 - threshold):
            status = "LOSS"
        elif wt_label == 0 and var_s > threshold:
            status = "GAIN"
        else:
            status = "NEUTRAL"

        results[go_term] = {
            "wt":     wt_label,
            "var":    round(var_s, 4),
            "status": status,
        }

        if wt_label == 1:
            annotated_var_scores.append(var_s)

    if annotated_var_scores:
        fitness = sum(annotated_var_scores) / len(annotated_var_scores)
    else:
        fitness = 0.0

    results["ContVAR_fitness"] = round(fitness, 4)

    return results
