use candle_core::{D, Error, Shape, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder, linear, ops::softmax};

pub struct SelfAttention {
    query_weights: Linear,
    key_weights: Linear,
    value_weights: Linear,
    dropout: Dropout,
}

impl SelfAttention {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        dropout: f32,
        vb: &VarBuilder,
    ) -> Result<Self, Error> {
        let query_weights = linear(dim_in, dim_out, vb.pp("queries"))?;
        let key_weights = linear(dim_in, dim_out, vb.pp("keys"))?;
        let value_weights = linear(dim_in, dim_out, vb.pp("values"))?;

        let dropout = Dropout::new(dropout);

        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            dropout,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let queries = self.query_weights.forward(input)?;
        let keys = self.key_weights.forward(input)?;
        let values = self.value_weights.forward(input)?;

        let attn_scores = queries.matmul(&keys.transpose(1, 2)?)?;
        let attn_scores = self.tril(&attn_scores)?;

        let dim_keys = *keys.dims().last().unwrap() as f64;
        let scale = dim_keys.sqrt();

        let scaled_scores = (attn_scores / scale)?;

        let attn_weights = softmax(&scaled_scores, D::Minus1)?;

        let attn_weights = self.dropout.forward(&attn_weights, true)?;

        let context_vector = attn_weights.matmul(&values)?;

        Ok(context_vector)
    }

    fn tril(&self, tensor: &Tensor) -> Result<Tensor, Error> {
        let n = tensor.dim(tensor.rank() - 2)?;
        let m = tensor.dim(tensor.rank() - 1)?;
        let (n_u32, m_u32) = (n as u32, m as u32);
        let device = tensor.device();

        let rows = Tensor::arange(0u32, n_u32, device)?.reshape((n, 1))?;
        let cols = Tensor::arange(0u32, m_u32, device)?.reshape((1, m))?;

        let mask = rows
            .broadcast_as((n, m))?
            .ge(&cols.broadcast_as((n, m))?)?
            .to_dtype(tensor.dtype())?
            .broadcast_as(tensor.shape())?;

        self.masked_fill(&mask, f32::NEG_INFINITY)
    }

    fn triu(&self, tensor: &Tensor) -> Result<Tensor, Error> {
        let n = tensor.dim(tensor.rank() - 2)?;
        let m = tensor.dim(tensor.rank() - 1)?;
        let (n_u32, m_u32) = (n as u32, m as u32);
        let device = tensor.device();

        let rows = Tensor::arange(0u32, n_u32, device)?.reshape((n, 1))?;
        let cols = Tensor::arange(0u32, m_u32, device)?.reshape((1, m))?;

        let mask = rows
            .broadcast_as((n, m))?
            .le(&cols.broadcast_as((n, m))?)?
            .to_dtype(tensor.dtype())?
            .broadcast_as(tensor.shape())?;

        self.masked_fill(&mask, f32::NEG_INFINITY)
    }

    fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor, Error> {
        let on_true = Tensor::new(value, mask.device())?.broadcast_as(mask.shape())?;
        mask.where_cond(&on_true, mask)
    }
}
