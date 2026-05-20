"""Lazy top-level exports for the ContVAR package."""

from importlib import import_module


__all__ = [
    "ProjectConfig",
    "setup_environment",
    "ensure_dms_triplets_unzipped",
    "DeepProteinGAT",
    "SemiHardMiningTripletLoss",
    "StandardTripletLoss",
    "get_loss_function",
    "train_pipeline",
    "visualize_tsne",
    "visualize_graph",
]


_LAZY_EXPORTS = {
    "ProjectConfig": ("contvar.config", "ProjectConfig"),
    "setup_environment": ("contvar.config", "setup_environment"),
    "ensure_dms_triplets_unzipped": ("contvar.config", "ensure_dms_triplets_unzipped"),
    "DeepProteinGAT": ("contvar.model", "DeepProteinGAT"),
    "SemiHardMiningTripletLoss": ("contvar.losses", "SemiHardMiningTripletLoss"),
    "StandardTripletLoss": ("contvar.losses", "StandardTripletLoss"),
    "get_loss_function": ("contvar.losses", "get_loss_function"),
    "train_pipeline": ("contvar.training", "train_pipeline"),
    "visualize_tsne": ("contvar.viz_tsne", "visualize_tsne"),
    "visualize_graph": ("contvar.viz_graph", "visualize_graph"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'contvar' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
