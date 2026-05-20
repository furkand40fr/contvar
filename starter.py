"""Starter CLI for ContVAR training."""

import argparse
import os

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# This is the single place where the application paths live.
# Edit these to point to your local files. The paths will be normalized to absolute paths before use.
# But you can also use absolute paths directly.
#
# - `protein_triplets_data/`
# - `embeddings_variable.h5`
# - `local_splits/...`
# - `semantic_similarity/`
# - checkpoints in the repository root
STARTER_PATHS = {

    # dms data root (protein_triplets_data/)
    "data_root": os.path.join(_REPO_ROOT, "protein_triplets_data"),

    # embedding path for dms mining (not needed for training, but used by t-SNE visualization)
    "embeddings_path": os.path.join(_REPO_ROOT, "embeddings_variable.h5"),

    # Paths for train/val/test splits.
    "dms_protein_split_json_path": os.path.join(
        _REPO_ROOT, "local_splits", "dms_protein_split.json"
    ),
    "go_protein_split_json_path": os.path.join(
        _REPO_ROOT, "local_splits", "phase0_protein_split_removed_graphless.json"
    ),
    "go_tsv_dir": os.path.join(_REPO_ROOT, "semantic_similarity"),

    # Prebuilt GO pretraining graphs.
    "go_prebuilt_graph_root": None,

    # Optional initialization checkpoint for GO phase-0 warm start.
    "go_phase0_init_checkpoint_path": None,

    # Paths for output models and visualizations.
    "go_phase0_best_model_path": os.path.join(
        _REPO_ROOT, "model_phase0_best_loss.pt"
    ),
    "go_phase0_last_model_path": os.path.join(_REPO_ROOT, "model_phase0_last.pt"),
    "stage2_best_model_path": os.path.join(_REPO_ROOT, "model_best_loss.pt"),
    "stage2_last_model_path": os.path.join(_REPO_ROOT, "model_last.pt"),
    "phase0_embeddings_export_path": os.path.join(
        _REPO_ROOT, "exports", "phase0_contvar_embeddings.h5"
    ),
    "dms_embeddings_export_path": os.path.join(
        _REPO_ROOT, "exports", "dms_variant_contvar_embeddings.h5"
    ),
    "tsne_save_dir": os.path.join(_REPO_ROOT, "visualizations"),
}

# Optional Colab defaults. Keep these here too so `config.py` does not need to
# carry a second copy of path configuration. ONLY used if the user explicitly selects the Colab option via command-line args.
COLAB_PATHS = {
    "data_root": "/content/content/content/protein_triplets_data",
    "data_zip": "/content/drive/MyDrive/ContVAR/protein_triplets_data_9march.zip",
    "embeddings_path": "/content/drive/MyDrive/ContVAR/embeddings_variable.h5",
    "extract_root": "/content/",
}


def _abs_or_none(path_value):
    """Normalize user-provided paths to absolute paths."""
    if path_value in (None, ""):
        return None
    return os.path.abspath(path_value)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="ContVAR starter CLI",
        epilog="Edit STARTER_PATHS in starter.py to change local file paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run_group = parser.add_argument_group("run options")
    run_group.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all protein graphs from scratch.",
    )

    return parser


def _resolve_paths():
    paths = {key: _abs_or_none(value) for key, value in STARTER_PATHS.items()}
    return paths


def _build_config_overrides(args, paths):
    overrides = {
        "go_phase0_epochs": 200,
        "go_tsv_dir": paths["go_tsv_dir"],
        "go_prebuilt_graph_root": paths["go_prebuilt_graph_root"],
        "go_phase0_init_checkpoint_path": paths["go_phase0_init_checkpoint_path"],
        "go_phase0_best_model_path": paths["go_phase0_best_model_path"],
        "go_phase0_last_model_path": paths["go_phase0_last_model_path"],
        "dms_protein_split_json_path": paths["dms_protein_split_json_path"],
        "go_protein_split_json_path": paths["go_protein_split_json_path"],
        "stage2_best_model_path": paths["stage2_best_model_path"],
        "stage2_last_model_path": paths["stage2_last_model_path"],
        "phase0_embeddings_export_path": paths["phase0_embeddings_export_path"],
        "dms_embeddings_export_path": paths["dms_embeddings_export_path"],
        "tsne_save_dir": paths["tsne_save_dir"],
    }

    return overrides


def _print_path_summary(paths, config_overrides):
    print("\nResolved ContVAR paths:")
    ordered_keys = [
        "data_root",
        "embeddings_path",
        "dms_protein_split_json_path",
        "go_protein_split_json_path",
        "go_tsv_dir",
        "go_prebuilt_graph_root",
        "go_phase0_init_checkpoint_path",
        "go_phase0_best_model_path",
        "go_phase0_last_model_path",
        "stage2_best_model_path",
        "stage2_last_model_path",
        "phase0_embeddings_export_path",
        "dms_embeddings_export_path",
        "tsne_save_dir",
    ]

    for key in ordered_keys:
        print(f"  {key}: {paths.get(key)}")

    phase0_epochs = config_overrides.get("go_phase0_epochs", "ProjectConfig default")
    print(f"  go_phase0_epochs: {phase0_epochs}")


def _build_runtime_config(config_overrides):
    from contvar.config import ProjectConfig

    cfg = ProjectConfig()
    for key, value in config_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _run_post_training_exports(model, mapper, processed_dir, config_overrides, paths, env):
    from contvar.export_embeddings import export_all_embeddings
    from contvar.viz_tsne import visualize_tsne

    cfg = _build_runtime_config(config_overrides)
    export_all_embeddings(
        model=model,
        cfg=cfg,
        device=env["device"],
        data_root=env["data_root"],
        embeddings_path=env["embeddings_path"],
        go_prebuilt_graph_root=paths["go_prebuilt_graph_root"],
        phase0_split_json_path=paths["go_protein_split_json_path"],
        dms_split_json_path=paths["dms_protein_split_json_path"],
        phase0_out_path=paths["phase0_embeddings_export_path"],
        dms_out_path=paths["dms_embeddings_export_path"],
        batch_size=32,
        force_dms_reprocess=False,
        include_dms_anchors=False,
    )

    print("\n=== Post-Training Visualization ===")
    visualize_tsne(
        model=model,
        mapper=mapper,
        processed_dir=processed_dir,
        split="val",
        device=env["device"],
        save_dir=paths["tsne_save_dir"],
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    paths = _resolve_paths()
    config_overrides = _build_config_overrides(args, paths)

    _print_path_summary(paths, config_overrides)

    from contvar.config import setup_environment
    from contvar.training import train_pipeline

    env = setup_environment(
        data_root=paths["data_root"],
        embeddings_path=paths["embeddings_path"],
    )

    model, mapper, processed_dir = train_pipeline(
        config=config_overrides,
        force=args.force,
        data_root=env["data_root"],
        embeddings_path=env["embeddings_path"],
        device=env["device"],
        data_zip=env.get("data_zip"),
    )

    if model is not None:
        _run_post_training_exports(
            model,
            mapper,
            processed_dir,
            config_overrides,
            paths,
            env,
        )


if __name__ == "__main__":
    main()
