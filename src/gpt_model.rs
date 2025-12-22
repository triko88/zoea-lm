use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{Int, Tensor, backend::Backend},
};

use crate::{
    gpt_config::GPTConfig,
    transformer::{TransformerBlock, TransformerConfig},
};

#[derive(Module, Debug)]
pub struct GPTModel<B: Backend> {
    token_embeddings: Embedding<B>,
    positional_embeddings: Embedding<B>,
    drop_embeddings: Dropout,
    blocks: Vec<TransformerBlock<B>>,
    final_normalization: LayerNorm<B>,
    output_head: Linear<B>,
}

impl<B: Backend> GPTModel<B> {
    pub fn new(config: GPTConfig, device: &B::Device) -> Self {
        let token_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.embedding_dims).init(device);
        let positional_embeddings =
            EmbeddingConfig::new(config.max_context_length, config.embedding_dims).init(device);

        let transformer_config = TransformerConfig {
            embedding_dims: config.embedding_dims,
            context_length: config.max_context_length,
            num_heads: config.num_heads,
            drop_rate: config.dropout,
            bias: config.bias,
        };

        let drop_embeddings = DropoutConfig::new(config.dropout).init();
        let blocks = vec![TransformerBlock::new(transformer_config, device); config.num_layers];
        let final_normalization = LayerNormConfig::new(config.embedding_dims).init(device);
        let output_head = LinearConfig::new(config.embedding_dims, config.vocab_size).init(device);

        Self {
            token_embeddings,
            positional_embeddings,
            drop_embeddings,
            blocks,
            final_normalization,
            output_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, num_tokens] = input.dims();
        let device = input.device();

        let token_embeddings = self.token_embeddings.forward(input);
        let positions: Tensor<B, 1, Int> = Tensor::arange(0..num_tokens as i64, &device);
        let positional_embeddings = self.positional_embeddings.forward(positions.unsqueeze());

        let embeddings = token_embeddings + positional_embeddings;
        let embeddings = self.drop_embeddings.forward(embeddings);

        let embeddings = self
            .blocks
            .iter()
            .fold(embeddings, |x, block| block.forward(x));

        let embeddings = self.final_normalization.forward(embeddings);

        self.output_head.forward(embeddings)
    }
}
