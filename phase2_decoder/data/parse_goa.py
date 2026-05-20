"""
GO annotation parsing and vocabulary building.

Source: goa_2025-12-04_swissprot_noiea.tsv
File format (no header, tab-separated):
    col 1  → UniProt accession   (P04637)
    col 4  → GO term ID          (GO:0003700)
    col 6  → Evidence code       (IDA, IBA, ND, ...)
    col 8  → Aspect              (F=MF, P=BP, C=CC)
    col 12 → Taxon               (taxon:9606)
"""

import json
from collections import defaultdict, Counter


EXCLUDE_EVIDENCE = {"ND", "IEA"}


def parse_goa_tsv(tsv_path: str, aspect: str = "F",
                  taxon_filter: str = None) -> dict[str, set]:
    """
    Returns: {uniprot_id: {GO:XXXXXXX, ...}}

    aspect       : "F" MF | "P" BP | "C" CC
    taxon_filter : None → all organisms | "taxon:9606" → human only
    """
    annotations: dict[str, set] = defaultdict(set)

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 13:
                continue

            uniprot_id = cols[1]
            go_id      = cols[4]
            evidence   = cols[6]
            asp        = cols[8]
            taxon      = cols[12].split("|")[0]

            if asp != aspect:
                continue
            if evidence in EXCLUDE_EVIDENCE:
                continue
            if taxon_filter and taxon != taxon_filter:
                continue

            annotations[uniprot_id].add(go_id)

    return dict(annotations)


def build_go_vocab(annotations: dict[str, set], min_freq: int = 10,
                   save_path: str = None) -> dict[str, int]:
    """
    GO term → integer index mapping.

    GO terms below min_freq are discarded.
    NULL_FUNCTION class is appended at the end.

    Returns: {"GO:0003700": 0, ..., "NULL_FUNCTION": N}
    """
    counts: Counter = Counter()
    for go_terms in annotations.values():
        counts.update(go_terms)

    filtered = [(go, cnt) for go, cnt in sorted(counts.items()) if cnt >= min_freq]
    vocab: dict[str, int] = {go: idx for idx, (go, cnt) in enumerate(filtered)}
    vocab["NULL_FUNCTION"] = len(vocab)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab
