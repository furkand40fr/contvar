# ContVAR

ContVAR trains a graph-based metric learning model for single amino acid variants (SAVs).
It uses triplets from the same protein:

- `anchor`: wild-type structure
- `positive`: benign variant
- `negative`: pathogenic variant

The model learns to pull benign variants closer to the anchor and push pathogenic variants farther away.

## Installation

```bash
pip install -e .
pip install torch torch-geometric graphein wandb biopython h5py scikit-learn matplotlib tqdm pandas numpy networkx MDAnalysis
```

## Expected data layout

Local defaults assume this repository layout:

```text
ContVAR/
|- starter.py
|- embeddings_variable.h5 (not needed if graphs for dms dataset already exist)
|- local_splits/
|  |- dms_protein_split.json
|  |- phase0_protein_split_removed_graphless.json
|- protein_triplets_data/
|  |- originals/
|  |- positives/
|  |- negatives/
|  ` - processed/
|-
`- semantic_similarity/
   |- semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv
   |- semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv
   `- semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv
`- a directory of prebuilt GO `.pt` graphs


## Starter CLI

`starter.py` is the main entry point for local runs.
It centralizes the runtime paths in one place through the `STARTER_PATHS` block near the top of the file.

### 1. Review or edit the default paths

Open [starter.py](starter.py) and update `STARTER_PATHS` if you want machine-specific defaults in one place.

The main workflow is:

```bash
python starter.py
```

A normal run automatically:

- saves the phase-0 checkpoints (models)
- saves the final stage checkpoints (models)
- exports the learned global ContVAR embeddings to H5
- generates the t-SNE visualizations

### 2. Run a DMS-only training job

If `STARTER_PATHS["go_prebuilt_graph_root"]` is left as `None`, the starter script automatically disables GO phase-0 pretraining and runs encoder DMS training only.

```bash
python starter.py
```

### 3. Run full training with GO phase-0 enabled

Set `STARTER_PATHS["go_prebuilt_graph_root"]` in `starter.py`, then run:

```bash
python starter.py
```

## Common CLI options

- `--force`: rebuild processed protein graphs from scratch

## Output files

By default, local runs write:

- `model_best_loss.pt`
- `model_last.pt`
- `exports/dms_variant_contvar_embeddings.h5`
- `model_phase0_best_loss.pt` when GO phase-0 is enabled
- `model_phase0_last.pt` when GO phase-0 is enabled
- `exports/phase0_contvar_embeddings.h5` when `go_prebuilt_graph_root` is configured
- `visualizations/`

## Notes

- GO phase-0 requires all of the following:
  `STARTER_PATHS["go_prebuilt_graph_root"]`, the GO TSV directory, and the GO split JSON.
- Encoder DMS training uses `protein_triplets_data`, `embeddings_variable.h5`, and the DMS split JSON.
- The exported H5 files contain the learned global graph embedding, not the local mutation-position embedding.
- The DMS export writes one embedding per variant file covered by `local_splits/dms_protein_split.json` across all families in the dataset.
- The training loop now reads checkpoint paths from configuration, and `starter.py` is the intended single place to edit local file paths.
