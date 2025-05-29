class ByT5Config:
    """Configuration class for ByT5-Small model"""

    def __init__(self):
        # Model architecture
        self.d_model = 1472  # Hidden size
        self.d_ff = 3584  # Feed-forward dimension
        self.encoder_layers = 12
        self.decoder_layers = 4
        self.num_heads = 8
        self.dropout_rate = 0.1

        # Vocabulary
        self.vocab_size = 259

        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

        # Sequence length
        self.max_length = 1024

        # Positional encoding
        self.use_relative_attention_bias = True
        self.relative_attention_num_buckets = 32

        # Pre-training settings
        self.mean_noise_span_length = 20

        # 2D-TPE
        self.lambda_entropy_loss = 0.01