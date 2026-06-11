class DecoderConfig:

    def __init__(self, aspect: str = "F", embedding_type: str = "concat"):
        # aspect: "F"=MF | "P"=BP | "C"=CC  (GO namespace)
        # embedding_type: "esm" | "contvar" | "contvar_full" | "concat" | "concat_full"
        self.aspect = aspect
        self.embedding_type = embedding_type
        asp = aspect.lower()

        # Encoder output dim & decoder architecture — depends on embedding type
        if embedding_type == "esm":
            self.encoder_output_dim = 1280
            self.hidden_dims        = [512, 1024, 512]
        elif embedding_type == "contvar":
            self.encoder_output_dim = 256
            self.hidden_dims        = [512, 1024, 512]
        elif embedding_type == "contvar_full":
            self.encoder_output_dim = 256
            self.hidden_dims        = [512, 1024, 512]
        elif embedding_type == "concat":
            # 1280 (ESM) + 256 (ContVAR GNN) = 1536
            self.encoder_output_dim = 1536
            self.hidden_dims        = [1024, 512]
        elif embedding_type == "concat_full":
            # 1280 (ESM) + 256 (ContVAR_Full GNN) = 1536
            self.encoder_output_dim = 1536
            self.hidden_dims        = [1024, 512]
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        self.dropout              = 0.2

        # GO classification
        self.min_go_freq          = 10
        self.null_function_weight = 3.0
        self.pos_weight_clamp     = 10.0

        # Training
        self.lr                   = 1e-3
        self.weight_decay         = 1e-05
        self.batch_size           = 256
        self.epochs               = 100
        self.early_stop_patience  = 20
        self.seed                 = 42
        self.eval_threshold       = "sweep"
        self.use_go_propagation   = False
        self.obo_path             = "go.obo"

        # File paths — aspect-specific
        self.goa_tsv              = "goa_2025-12-04_swissprot_noiea.tsv"
        self.esm_h5               = "esm2_t33_650M_UR50D_protein_embedding.h5"
        self.contvar_h5           = "go_pretraining_contvar_embeddings.h5"
        self.contvar_full_h5      = "stage2_best_pretraining_protein_embeddings.h5"
        self.go_vocab_json        = f"go_vocab_{asp}.json"
        self.decoder_checkpoint   = f"decoder_best_{embedding_type}_{asp}.pt"
        self.uniref_tsv           = "protein_uniref50.tsv"
        self.split_json           = "phase0_go_split.json"

        # Weights & Biases
        self.wandb_project        = "ContVAR-Project"
        self.wandb_entity         = "canerayvaz-hacettepe-university"
        self.wandb_run_name       = f"decoder-{embedding_type}-{asp}"
        self.wandb_api_key        = "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"
