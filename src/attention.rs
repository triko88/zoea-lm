use core::num;
use std::f32::NEG_INFINITY;

use burn::{
    Tensor,
    nn::{Dropout, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Bool, Shape, activation},
};

use crate::attention;

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
    mask: Tensor<B, 2, Bool>,
    dropout: Dropout,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(config: MultiHeadAttentionConfig) -> Self {
        let linear_config = LinearConfig::new(config.input_dimensions, config.output_dimensions)
            .with_bias(config.with_bias);

        let device = B::Device::default();

        let ones = Tensor::<B, 2>::ones([config.context_length, config.context_length], &device);
        let mask = ones.triu(1).bool();

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

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut dims = x.shape().dims::<2>().to_vec();
        let num_tokens = dims[1];

        dims.push(self.head_dimesnion);

        let shape = Shape { dims: dims };

        let query = self.query_weights.forward(x.clone()).reshape(shape.clone());
        let key = self.key_weights.forward(x.clone()).reshape(shape.clone());
        let value = self.value_weights.forward(x).reshape(shape);

        let query = query.swap_dims(1, 2);
        let key = key.swap_dims(1, 2);
        let value = value.swap_dims(1, 2);

        let scale = (self.head_dimesnion as f64).sqrt();

        let mut attn_scores = query.matmul(key.swap_dims(2, 3)) / scale;
        let mask = self.mask.clone().slice([..num_tokens, ..num_tokens]);

        attn_scores = attn_scores.mask_fill(mask, NEG_INFINITY);

        let attn_weights = activation::softmax(attn_scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vector = attn_weights.matmul(value).swap_dims(1, 2).flatten(2, 3);

        self.out_projection.forward(context_vector)
    }
}
