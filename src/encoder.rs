use candle_core::{DType, Error, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};

use crate::dataloader::DataLoader;

pub struct Encoder {
    vocab_size: usize,
    context_length: usize,
    output_dimensions: usize,
}

impl Encoder {
    pub fn new(vocab_size: usize, context_length: usize, output_dimensions: usize) -> Self {
        Self {
            vocab_size,
            context_length,
            output_dimensions,
        }
    }

    fn build_embeddings(&self, builder: &VarBuilder) -> Result<(Embedding, Embedding), Error> {
        let token_embedding_layer = embedding(
            self.vocab_size,
            self.output_dimensions,
            builder.pp("token_embeddings"),
        )?;
        let pos_embedding_layer = embedding(
            self.context_length,
            self.output_dimensions,
            builder.pp("pos_embeddings"),
        )?;

        Ok((token_embedding_layer, pos_embedding_layer))
    }

    pub fn get_input_embeddings(&self, data_loader: DataLoader) -> Result<Tensor, Error> {
        let (token_layer, pos_layer) = self.build_embeddings(&data_loader.builder)?;
        let mut result: Vec<Tensor> = Vec::new();
        let device = data_loader.builder.device().clone();

        for batch in data_loader {
            let (x, _) = batch;
            let token_embeddings = token_layer.forward(&x)?;

            let len = x.dim(1)? as i64;
            let positions = Tensor::arange(0i64, len, &device)?
                .to_dtype(DType::I64)?
                .unsqueeze(0)?;
            let position_embeddings = pos_layer
                .forward(&positions)?
                .expand(token_embeddings.shape())?;

            let input_embeddings = (token_embeddings + position_embeddings)?;

            result.push(input_embeddings);
        }

        Tensor::cat(&result, 0)
    }
}
