use core::f32;

use candle_core::{D, DType, Error, IndexOp, Shape, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder, linear, ops::softmax};

pub struct SelfAttention {
    query_weights: Linear,
    key_weights: Linear,
    value_weights: Linear,
    dropout: Dropout,
    mask: Tensor,
}

impl SelfAttention {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        context_length: usize,
        dropout: f32,
        vb: &VarBuilder,
    ) -> Result<Self, Error> {
        let query_weights = linear(dim_in, dim_out, vb.pp("queries"))?;
        let key_weights = linear(dim_in, dim_out, vb.pp("keys"))?;
        let value_weights = linear(dim_in, dim_out, vb.pp("values"))?;

        let dropout = Dropout::new(dropout);

        let mask = Tensor::triu2(context_length, DType::U8, vb.device())?;

        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            dropout,
            mask,
        })
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> Result<Tensor, Error> {
        let num_tokens = input.dims()[1];

        let queries = self.query_weights.forward(input)?;
        let keys = self.key_weights.forward(input)?;
        let values = self.value_weights.forward(input)?;

        let attn_scores = queries.matmul(&keys.transpose(1, 2)?)?;

        let mask = self.mask.i((..num_tokens, ..num_tokens))?;
        let mask = mask.broadcast_as(attn_scores.shape())?;

        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, input.device())?.broadcast_as(attn_scores.shape())?;

        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let dim_keys = *keys.dims().last().unwrap() as f64;
        let scale = dim_keys.sqrt();

        let scaled_scores = (attn_scores / scale)?;

        let attn_weights = softmax(&scaled_scores, D::Minus1)?;

        let attn_weights = self.dropout.forward(&attn_weights, train)?;

        let context_vector = attn_weights.matmul(&values)?;

        Ok(context_vector)
    }
}
