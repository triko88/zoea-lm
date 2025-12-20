use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::backend::Backend,
};

use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    activation: Gelu,
    linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(embeddeding_dimesnions: usize, device: &B::Device) -> Self {
        let hidden_dimensions = 4 * embeddeding_dimesnions;

        let linear1 = LinearConfig::new(embeddeding_dimesnions, hidden_dimensions).init(device);
        let linear2 = LinearConfig::new(hidden_dimensions, embeddeding_dimesnions).init(device);
        let activation = Gelu::new();

        Self {
            linear1,
            activation,
            linear2,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let hidden = self.linear1.forward(x);
        let activated = self.activation.forward(hidden);
        self.linear2.forward(activated)
    }
}

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub embedding_dims: usize,
    pub context_length: usize,
    pub num_heads: usize,
    pub drop_rate: f64,
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    feed_forward: FeedForward<B>,
    normal1: LayerNorm<B>,
    normal2: LayerNorm<B>,
    drop_shortcut: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: TransformerConfig, device: B::Device) -> Self {
        let attn_condig = MultiHeadAttentionConfig::new()
            .with_input_dimensions(config.embedding_dims)
            .with_output_dimensions(config.embedding_dims)
            .with_context_length(config.context_length)
            .with_num_heads(config.num_heads)
            .with_dropout(config.drop_rate)
            .with_bias(config.bias);

        let attention = MultiHeadAttention::new(attn_condig);
        let feed_forward = FeedForward::new(config.embedding_dims, &device);

        let normal1 = LayerNormConfig::new(config.embedding_dims).init(&device);
        let normal2 = LayerNormConfig::new(config.embedding_dims).init(&device);

        let drop_shortcut = DropoutConfig::new(config.drop_rate).init();

        Self {
            attention,
            feed_forward,
            normal1,
            normal2,
            drop_shortcut,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = input.clone();

        // Attention block with residual connection
        let input = self.normal1.forward(input);
        let input = self.attention.forward(input);
        let input = self.drop_shortcut.forward(input);
        let input = input + shortcut;

        // Feed forward block with residual connection
        let shortcut = input.clone();
        let input = self.normal2.forward(input);
        let input = self.feed_forward.forward(input);
        let input = self.drop_shortcut.forward(input);

        input + shortcut
    }
}
