use burn::config::Config;

#[derive(Config, Debug)]
pub struct GPTConfig {
    #[config(default = 50257)]
    pub vocab_size: usize,

    #[config(default = 1024)]
    pub max_context_length: usize,

    #[config(default = 12)]
    pub num_heads: usize,

    #[config(default = 12)]
    pub num_layers: usize,

    #[config(default = 768)]
    pub embedding_dims: usize,

    #[config(default = 0.0)]
    pub dropout: f64,

    #[config(default = false)]
    pub bias: bool,
}

impl GPTConfig {
    pub fn gpt2_124m() -> Self {
        Self::new()
            .with_vocab_size(50257)
            .with_max_context_length(1024)
            .with_num_heads(12)
            .with_num_layers(24)
            .with_embedding_dims(768)
            .with_dropout(0.0)
            .with_bias(false)
    }

    pub fn tiny() -> Self {
        Self::new()
            .with_vocab_size(50257)
            .with_max_context_length(256)
            .with_num_heads(4)
            .with_num_layers(4)
            .with_embedding_dims(128)
            .with_dropout(0.0)
            .with_bias(false)
    }
}
