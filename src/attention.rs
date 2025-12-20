use std::f32::NEG_INFINITY;

use burn::{
    Tensor,
    nn::{Dropout, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Bool, Shape, activation},
};

pub struct MultiHeadAttentionConfig {
    pub input_dimensions: usize,
    pub output_dimensions: usize,
    pub context_length: usize,
    pub num_heads: usize,
    pub drop_out: f64,
    pub with_bias: bool,
}

pub struct MultiHeadAttention<B: Backend> {
    query_weights: Linear<B>,
    key_weights: Linear<B>,
    value_weights: Linear<B>,
    head_dimesnion: usize,
    out_projection: Linear<B>,
    mask: Tensor<B, 4, Bool>,
    dropout: Dropout,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(config: MultiHeadAttentionConfig) -> Self {
        let linear_config = LinearConfig::new(config.input_dimensions, config.output_dimensions)
            .with_bias(config.with_bias);

        let device = B::Device::default();

        let ones = Tensor::<B, 2>::ones([config.context_length, config.context_length], &device);
        let mask =
            ones.triu(1)
                .bool()
                .reshape([1, 1, config.context_length, config.context_length]);

        let out_projection =
            LinearConfig::new(config.output_dimensions, config.input_dimensions).init(&device);

        Self {
            query_weights: linear_config.init(&device),
            key_weights: linear_config.init(&device),
            value_weights: linear_config.init(&device),
            head_dimesnion: config.output_dimensions / config.num_heads,
            out_projection,
            mask,
            dropout: Dropout {
                prob: config.drop_out,
            },
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, num_tokens, d_model] = x.shape().dims();

        let num_heads = d_model / self.head_dimesnion;

        let shape = Shape {
            dims: vec![batch_size, num_tokens, num_heads, self.head_dimesnion],
        };

        let query: Tensor<B, 4> = self.query_weights.forward(x.clone()).reshape(shape.clone());
        let key: Tensor<B, 4> = self.key_weights.forward(x.clone()).reshape(shape.clone());
        let value: Tensor<B, 4> = self.value_weights.forward(x).reshape(shape);

        let query = query.swap_dims(1, 2);
        let key = key.swap_dims(1, 2);
        let value = value.swap_dims(1, 2);

        let scale = (self.head_dimesnion as f64).sqrt();

        let mut attn_scores = query.matmul(key.swap_dims(2, 3)) / scale;
        let mask: Tensor<B, 4, Bool> =
            self.mask
                .clone()
                .slice([..1, ..1, ..num_tokens, ..num_tokens]);

        attn_scores = attn_scores.mask_fill(mask, NEG_INFINITY);

        let attn_weights = activation::softmax(attn_scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vector = attn_weights.matmul(value).swap_dims(1, 2).flatten(2, 3);

        self.out_projection.forward(context_vector)
    }
}
