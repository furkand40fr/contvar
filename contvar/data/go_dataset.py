import os
import random
from typing import Dict, List, Tuple, Optional, Literal, Set

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from contvar.config import ProjectConfig


class GOSemanticTripletDataset(Dataset):
    """
    Dataset for GO semantic similarity pretraining (phase 0).

    Loads protein graphs only from prebuilt PyG ``.pt`` files under
    ``prebuilt_graph_root`` (no CIF / on-the-fly graph construction).

    Each sample is an (anchor, positive, negative) triplet from semantic
    similarity TSVs:

    - positives: sim >= 0.8
    - negatives: sim <= 0.2, split into two bins for balanced hard negatives:
        * bin_low:   [0.0, 0.1)
        * bin_mid:   [0.1, 0.2]
    """

    _global_pt_path_index: Dict[str, Dict[str, str]] = {}
    _global_pt_issue_messages: Dict[str, str] = {}
    _global_reported_pt_issues: Set[str] = set()
    _max_example_ids_to_print = 10
    _max_runtime_issue_examples = 3

    @classmethod
    def warm_prebuilt_index(cls, prebuilt_graph_root: str) -> int:
        """Build the .pt path index for this root; return number of indexed protein prefixes."""
        if not prebuilt_graph_root or not os.path.isdir(prebuilt_graph_root):
            return 0
        root_key = os.path.abspath(prebuilt_graph_root)
        if root_key in cls._global_pt_path_index:
            return len(cls._global_pt_path_index[root_key])
        index: Dict[str, str] = {}
        for root, _, files in os.walk(prebuilt_graph_root):
            for fname in files:
                if not fname.lower().endswith(".pt"):
                    continue
                base = os.path.splitext(fname)[0].lower()
                prefix = base.split("_", 1)[0]
                if prefix:
                    index[prefix] = os.path.join(root, fname)
        cls._global_pt_path_index[root_key] = index
        return len(index)

    def __init__(
        self,
        tsv_path: str,
        ontology: str,
        config: ProjectConfig,
        prebuilt_graph_root: str,
        sim_col: Optional[str] = None,
        pos_threshold: float = 0.8,
        neg_low: float = 0.0,
        neg_mid: float = 0.1,
        neg_high: float = 0.2,
        phase0_split: Optional[Literal["train", "val", "test"]] = None,
        protein_to_split: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        if not prebuilt_graph_root or not os.path.isdir(prebuilt_graph_root):
            raise FileNotFoundError(
                f"[GO-{ontology}] prebuilt_graph_root must be an existing directory: "
                f"{prebuilt_graph_root!r}"
            )

        self.tsv_path = tsv_path
        self.ontology = ontology
        self.config = config
        self.prebuilt_graph_root = prebuilt_graph_root

        if sim_col is None:
            sim_col = f"sim_{ontology}"
        self.sim_col = sim_col

        self.pos_threshold = pos_threshold
        self.neg_low = neg_low
        self.neg_mid = neg_mid
        self.neg_high = neg_high
        self.phase0_split = phase0_split
        self.protein_to_split = protein_to_split

        print(
            f"[GO-{ontology}] Phase-0 graphs: prebuilt .pt only from {prebuilt_graph_root}"
        )

        self.triplets: List[Tuple[str, str, str]] = []
        self.graph_cache: Dict[str, Data] = {}

        self._parse_tsv()

        if self.phase0_split and self.protein_to_split:
            from contvar.go_identity_split import filter_triplets_by_split

            before = len(self.triplets)
            self.triplets = filter_triplets_by_split(
                self.triplets, self.protein_to_split, self.phase0_split
            )
            print(
                f"[GO-{self.ontology}] Split={self.phase0_split}: "
                f"{before:,} -> {len(self.triplets):,} triplets (identity filter)"
            )

        before = len(self.triplets)
        available_ids = self._get_available_prebuilt_ids()
        missing_ids = self._collect_missing_prebuilt_ids(self.triplets, available_ids)
        self.triplets = [
            (a, p, n)
            for (a, p, n) in self.triplets
            if a.lower() in available_ids
            and p.lower() in available_ids
            and n.lower() in available_ids
        ]
        if missing_ids:
            example_ids = ", ".join(
                missing_ids[: self._max_example_ids_to_print]
            )
            remaining = len(missing_ids) - self._max_example_ids_to_print
            more_suffix = f" (+{remaining} more)" if remaining > 0 else ""
            print(
                f"[GO-{self.ontology}] Missing prebuilt .pt graphs for "
                f"{len(missing_ids):,} proteins; triplets containing them were dropped. "
                f"Examples: {example_ids}{more_suffix}"
            )
        print(
            f"[GO-{self.ontology}] Prebuilt .pt filter: "
            f"{before:,} -> {len(self.triplets):,} triplets "
            f"(available proteins={len(available_ids):,})"
        )

    def _parse_tsv(self):
        if not os.path.exists(self.tsv_path):
            print(f"[GO-{self.ontology}] TSV not found: {self.tsv_path} (skipping)")
            return

        import csv

        anchors: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

        with open(self.tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                anchor_id = row["anchor"]
                cand_id = row["candidate"]
                try:
                    sim_val = float(row[self.sim_col])
                except (KeyError, ValueError):
                    continue

                if anchor_id not in anchors:
                    anchors[anchor_id] = {
                        "pos": [],
                        "neg_low": [],
                        "neg_mid": [],
                    }

                if sim_val >= self.pos_threshold:
                    anchors[anchor_id]["pos"].append((cand_id, sim_val))
                elif self.neg_low <= sim_val < self.neg_mid:
                    anchors[anchor_id]["neg_low"].append((cand_id, sim_val))
                elif self.neg_mid <= sim_val <= self.neg_high:
                    anchors[anchor_id]["neg_mid"].append((cand_id, sim_val))

        for anchor_id, buckets in anchors.items():
            pos_list = buckets["pos"]
            neg_low = buckets["neg_low"]
            neg_mid = buckets["neg_mid"]

            if not pos_list:
                continue
            if not (neg_low or neg_mid):
                continue

            for pos_id, _ in pos_list:
                has_low = len(neg_low) > 0
                has_mid = len(neg_mid) > 0

                if has_low and has_mid:
                    if random.random() < 0.5:
                        neg_id, _ = random.choice(neg_low)
                    else:
                        neg_id, _ = random.choice(neg_mid)
                elif has_low:
                    neg_id, _ = random.choice(neg_low)
                elif has_mid:
                    neg_id, _ = random.choice(neg_mid)
                else:
                    continue

                self.triplets.append((anchor_id, pos_id, neg_id))

        print(
            f"[GO-{self.ontology}] Parsed {len(self.triplets):,} triplets from {self.tsv_path}"
        )

    def _id_to_prebuilt_graph_path(self, protein_id: str) -> Optional[str]:
        pid_lower = protein_id.lower()
        root_key = os.path.abspath(self.prebuilt_graph_root)
        index = self.__class__._global_pt_path_index.get(root_key)
        if index is None:
            index = {}
            for root, _, files in os.walk(self.prebuilt_graph_root):
                for fname in files:
                    if not fname.lower().endswith(".pt"):
                        continue
                    base = os.path.splitext(fname)[0].lower()
                    prefix = base.split("_", 1)[0]
                    if prefix:
                        index[prefix] = os.path.join(root, fname)
            self.__class__._global_pt_path_index[root_key] = index

        return index.get(pid_lower)

    def _get_available_prebuilt_ids(self) -> set:
        root_key = os.path.abspath(self.prebuilt_graph_root)
        index = self.__class__._global_pt_path_index.get(root_key)
        if index is None:
            self._id_to_prebuilt_graph_path("__index_warmup__")
            index = self.__class__._global_pt_path_index.get(root_key, {})
        return set(index.keys())

    def _collect_missing_prebuilt_ids(
        self, triplets: List[Tuple[str, str, str]], available_ids: set
    ) -> List[str]:
        missing_ids: List[str] = []
        seen_missing: Set[str] = set()
        for anchor_id, pos_id, neg_id in triplets:
            for protein_id in (anchor_id, pos_id, neg_id):
                protein_id_lower = protein_id.lower()
                if protein_id_lower in available_ids or protein_id_lower in seen_missing:
                    continue
                seen_missing.add(protein_id_lower)
                missing_ids.append(str(protein_id).upper())
        return missing_ids

    @classmethod
    def _remember_pt_issue(cls, prebuilt_path: str, issue_message: str):
        if prebuilt_path not in cls._global_pt_issue_messages:
            cls._global_pt_issue_messages[prebuilt_path] = issue_message
        if prebuilt_path in cls._global_reported_pt_issues:
            return
        cls._global_reported_pt_issues.add(prebuilt_path)
        print(
            f"[GO-phase0] Problem loading prebuilt graph {prebuilt_path}: "
            f"{cls._global_pt_issue_messages[prebuilt_path]}"
        )

    def _describe_graph_issue(self, protein_id: str) -> Optional[str]:
        prebuilt_path = self._id_to_prebuilt_graph_path(protein_id)
        if prebuilt_path is None:
            return f"{protein_id}: no indexed .pt file"

        issue_message = self.__class__._global_pt_issue_messages.get(prebuilt_path)
        if issue_message:
            return f"{protein_id}: {issue_message} ({prebuilt_path})"
        return f"{protein_id}: indexed at {prebuilt_path} but could not be loaded"

    def _load_prebuilt_graph(self, protein_id: str) -> Optional[Data]:
        prebuilt_path = self._id_to_prebuilt_graph_path(protein_id)
        if prebuilt_path is None:
            return None
        try:
            data = torch.load(prebuilt_path, weights_only=False)
            if isinstance(data, Data):
                return data
            self._remember_pt_issue(
                prebuilt_path,
                f"expected torch_geometric.data.Data, got {type(data).__name__}",
            )
            return None
        except Exception as exc:
            self._remember_pt_issue(
                prebuilt_path,
                f"{type(exc).__name__}: {exc}",
            )
            return None

    def _get_graph(self, protein_id: str) -> Optional[Data]:
        if protein_id in self.graph_cache:
            return self.graph_cache[protein_id]

        g = self._load_prebuilt_graph(protein_id)
        if g is not None:
            self.graph_cache[protein_id] = g
        return g

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        max_attempts = 20
        attempted_issues: List[str] = []
        seen_issues: Set[str] = set()
        for _ in range(max_attempts):
            anchor_id, pos_id, neg_id = self.triplets[idx]

            g_a = self._get_graph(anchor_id)
            g_p = self._get_graph(pos_id)
            g_n = self._get_graph(neg_id)

            if g_a is not None and g_p is not None and g_n is not None:
                return g_a, g_p, g_n

            for protein_id, graph_obj in (
                (anchor_id, g_a),
                (pos_id, g_p),
                (neg_id, g_n),
            ):
                if graph_obj is not None:
                    continue
                issue = self._describe_graph_issue(protein_id)
                if issue and issue not in seen_issues:
                    seen_issues.add(issue)
                    attempted_issues.append(issue)

            idx = random.randint(0, len(self.triplets) - 1)

        issue_summary = ""
        if attempted_issues:
            examples = attempted_issues[: self._max_runtime_issue_examples]
            remaining = len(attempted_issues) - len(examples)
            more_suffix = f" (+{remaining} more)" if remaining > 0 else ""
            issue_summary = (
                " Recent graph issues: "
                + "; ".join(examples)
                + more_suffix
                + "."
            )
        raise RuntimeError(
            f"[GO-{self.ontology}] Failed to load .pt graphs after {max_attempts} attempts. "
            f"Check prebuilt files for IDs in {self.tsv_path} under {self.prebuilt_graph_root}."
            f"{issue_summary}"
        )
