use candle_core::{D, DType, Device, Error, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, ops::softmax};

pub struct SelfAttention {
    query_weights: Linear,
    key_weights: Linear,
    value_weights: Linear,
}

impl SelfAttention {
    pub fn new(dim_in: usize, dim_out: usize, vb: &VarBuilder) -> Result<Self, Error> {
        let query_weights = linear(dim_in, dim_out, vb.pp("queries"))?;
        let key_weights = linear(dim_in, dim_out, vb.pp("keys"))?;
        let value_weights = linear(dim_in, dim_out, vb.pp("values"))?;

        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
        })
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        println!("initialising...");

        let queries = self.query_weights.forward(input)?;
        let keys = self.key_weights.forward(input)?;
        let values = self.value_weights.forward(input)?;

        println!("getting scores...");
        let attn_scores = queries.matmul(&keys.t()?)?;

        let dim_keys = *keys.dims().last().unwrap() as f64;
        let scale = dim_keys.sqrt();

        println!("scaling scores...");
        let scaled_scores = (attn_scores / scale)?;

        let attn_weights = softmax(&scaled_scores, D::Minus1)?;
        println!("scaling weights...");

        let context_vector = attn_weights.matmul(&values)?;

        println!("done...");
        Ok(context_vector)
    }
}
