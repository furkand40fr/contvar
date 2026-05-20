import os
import glob
import json


class DmsProteinSplitError(Exception):
    """Raised when the fixed protein-level DMS split configuration is invalid."""


class TripletDataPathMapper:
    """Maps protein file structure to anchor-positive-negative triplets.

    Stage-2 DMS training always uses a fixed protein-level split JSON where each
    protein family basename maps to one of: ``train``, ``val``, ``test``.
    All variants for a reserved family stay in the same split.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(self, root_dir, split_json_path):
        """
        Args:
            root_dir: Path to protein_triplets_data directory.
            split_json_path: Path to fixed protein-level JSON split file.
        """
        self.root_dir = root_dir
        self.split_json_path = split_json_path

        # All protein family data (full, before splitting)
        self.triplets = []

        # Per-split triplets at the family level
        self.train_triplets = []
        self.val_triplets = []
        self.test_triplets = []
        self.family_to_protein_id = {}

        self._map_data()
        self._load_fixed_split()

    def _map_data(self):
        """Discover all protein families and their variant files."""
        originals = glob.glob(os.path.join(self.root_dir, 'originals', "*.cif"))

        for anchor in originals:
            prot_id = os.path.splitext(os.path.basename(anchor))[0]
            pos_dir = os.path.join(self.root_dir, 'positives', prot_id)
            neg_dir = os.path.join(self.root_dir, 'negatives', prot_id)

            p_files = sorted(glob.glob(os.path.join(pos_dir, "*.cif")))
            n_files = sorted(glob.glob(os.path.join(neg_dir, "*.cif")))

            if p_files and n_files:
                self.triplets.append({
                    'anchor': anchor,
                    'positives': p_files,
                    'negatives': n_files,
                    'protein_id': prot_id
                })

        # Sort by protein_id for deterministic ordering
        self.triplets.sort(key=lambda t: t['protein_id'])
        print(f"Found {len(self.triplets)} protein families")

    def _load_json_bundle(self):
        """Read and validate the protein-level split JSON."""
        if not self.split_json_path:
            raise DmsProteinSplitError(
                "ProjectConfig.dms_protein_split_json_path is not set."
            )
        if not os.path.isfile(self.split_json_path):
            raise DmsProteinSplitError(
                f"DMS protein split JSON not found: {self.split_json_path}"
            )

        try:
            with open(self.split_json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DmsProteinSplitError(
                f"Failed to parse DMS protein split JSON at {self.split_json_path}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise DmsProteinSplitError(
                f"Could not read DMS protein split JSON at {self.split_json_path}: {exc}"
            ) from exc

        family_to_split = data.get("family_to_split")
        if not isinstance(family_to_split, dict) or not family_to_split:
            raise DmsProteinSplitError(
                f"DMS protein split JSON must contain a non-empty top-level "
                f"'family_to_split' mapping: {self.split_json_path}"
            )

        invalid_labels = {
            family: split_name
            for family, split_name in family_to_split.items()
            if split_name not in self.VALID_SPLITS
        }
        if invalid_labels:
            sample_family, sample_split = next(iter(invalid_labels.items()))
            raise DmsProteinSplitError(
                f"Invalid split label {sample_split!r} for family {sample_family!r}. "
                f"Allowed labels: {', '.join(self.VALID_SPLITS)}."
            )

        family_to_protein_id = data.get("family_to_protein_id", {})
        if family_to_protein_id and not isinstance(family_to_protein_id, dict):
            raise DmsProteinSplitError(
                f"'family_to_protein_id' must be a mapping when provided: {self.split_json_path}"
            )

        return family_to_split, family_to_protein_id

    def _validate_split_coverage(self, family_to_split, family_to_protein_id):
        """Ensure JSON keys match exact family basenames present on disk."""
        dataset_families = {t['protein_id'] for t in self.triplets}
        split_families = set(family_to_split)

        missing_in_json = sorted(dataset_families - split_families)
        extra_in_json = sorted(split_families - dataset_families)
        if missing_in_json or extra_in_json:
            details = []
            if missing_in_json:
                details.append(
                    "missing from JSON: " + ", ".join(repr(x) for x in missing_in_json[:5])
                )
            if extra_in_json:
                details.append(
                    "not found in protein_triplets_data: "
                    + ", ".join(repr(x) for x in extra_in_json[:5])
                )
            raise DmsProteinSplitError(
                "DMS protein split family names must match the exact anchor basenames "
                "under protein_triplets_data/originals. " + " | ".join(details)
            )

        if family_to_protein_id:
            accession_families = set(family_to_protein_id)
            missing_accessions = sorted(dataset_families - accession_families)
            extra_accessions = sorted(accession_families - dataset_families)
            if missing_accessions or extra_accessions:
                details = []
                if missing_accessions:
                    details.append(
                        "missing accession entries: "
                        + ", ".join(repr(x) for x in missing_accessions[:5])
                    )
                if extra_accessions:
                    details.append(
                        "extra accession entries: "
                        + ", ".join(repr(x) for x in extra_accessions[:5])
                    )
                raise DmsProteinSplitError(
                    "family_to_protein_id keys must match the same family basenames as "
                    "family_to_split. " + " | ".join(details)
                )

    def _load_fixed_split(self):
        """Assign entire protein families to train/val/test from the JSON bundle."""
        family_to_split, family_to_protein_id = self._load_json_bundle()
        self._validate_split_coverage(family_to_split, family_to_protein_id)

        self.train_triplets = []
        self.val_triplets = []
        self.test_triplets = []
        self.family_to_protein_id = family_to_protein_id or {}

        split_to_triplets = {
            "train": self.train_triplets,
            "val": self.val_triplets,
            "test": self.test_triplets,
        }

        for t in self.triplets:
            family_name = t['protein_id']
            split_name = family_to_split[family_name]
            record = dict(t)
            if self.family_to_protein_id:
                record['accession'] = self.family_to_protein_id.get(family_name)
            split_to_triplets[split_name].append(record)

        print(f"Loaded fixed DMS protein split from {self.split_json_path}")
        for split_name, split_triplets in split_to_triplets.items():
            total_pos = sum(len(t['positives']) for t in split_triplets)
            total_neg = sum(len(t['negatives']) for t in split_triplets)
            print(
                f"  {split_name.capitalize():<5} families: {len(split_triplets):>3} | "
                f"variants: {total_pos} positives, {total_neg} negatives"
            )

    def get_split(self, split='train'):
        """Get triplets for a specific split."""
        if split == 'train':
            return self.train_triplets
        elif split == 'val':
            return self.val_triplets
        elif split == 'test':
            return self.test_triplets
        else:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'"
            )
