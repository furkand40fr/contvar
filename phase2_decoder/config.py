import os


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PHASE2_ROOT = os.path.join(_REPO_ROOT, "phase2_decoder")
_EXPORT_ROOT = os.path.join(_PHASE2_ROOT, "exports")


class DecoderConfig:

    def __init__(self, aspect: str = "F"):
        # aspect: "F"=MF | "P"=BP | "C"=CC  (GO namespace)
        self.aspect = aspect
        asp = aspect.lower()

        # Encoder
        # ContVAR exports 256-d global graph embeddings for the decoder.
        self.encoder_output_dim = 256

        # Decoder architecture
        self.hidden_dims = [512, 1024, 512]
        self.dropout = 0.3

        # GO classification
        self.min_go_freq = 10
        self.null_function_weight = 3.0
        self.pos_weight_clamp = 20.0

        # Training
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 256
        self.epochs = 100
        self.early_stop_patience = 10
        self.seed = 42
        self.eval_threshold = 0.95

        # File paths - aspect-specific
        self.goa_tsv = os.path.join(_REPO_ROOT, "goa_2025-12-04_swissprot_noiea.tsv")
        self.embeddings_h5 = os.path.join(_EXPORT_ROOT, "phase0_contvar_embeddings.h5")
        self.go_vocab_json = os.path.join(_PHASE2_ROOT, f"go_vocab_{asp}.json")
        self.decoder_checkpoint = os.path.join(_PHASE2_ROOT, f"decoder_best_{asp}.pt")
        self.uniref_tsv = os.path.join(_REPO_ROOT, "protein_uniref50.tsv")
        self.split_json = os.path.join(
            _REPO_ROOT,
            "local_splits",
            "phase0_protein_split_removed_graphless.json",
        )

        # Weights & Biases
        self.wandb_project = "ContVAR-Project"
        self.wandb_entity = "canerayvaz-hacettepe-university"
        self.wandb_run_name = f"decoder-{asp}"
        self.wandb_api_key = None
