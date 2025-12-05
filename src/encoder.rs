use candle_nn::{Embedding, Module, VarBuilder, embedding};
use candle_core::{DType, Device, Error, Tensor};

use crate::dataloader::DataLoader;

pub struct Encoder {
    vocab_size:         usize,
    context_length:     usize,
    output_dimensions:   usize,
    data_type:          DType,
    device:             Device,
}

impl Encoder {
    pub fn new(vocab_size: usize, context_length: usize, output_dimensions: usize, device: Device) -> Self {
        Self {
            vocab_size,
            context_length,
            output_dimensions,
            data_type: DType::F32,
            device,
        }
    }

    fn build_embeddings(&self) -> Result<(Embedding, Embedding), Error> {
        let vb = VarBuilder::zeros(self.data_type.clone(), &self.device);

        let token_embedding_layer = embedding(self.vocab_size, self.output_dimensions, vb.pp("token_embeddings"))?;
        let pos_embedding_layer = embedding(self.context_length,self.output_dimensions, vb.pp("pos_embeddings"))?;
        
        Ok((token_embedding_layer, pos_embedding_layer))
    }
    
    pub fn get_input_embeddings(&self, data_loader: DataLoader) -> Result<Vec<Tensor>, Error> {
        let (token_layer, pos_layer) = self.build_embeddings()?;
        let mut result: Vec<Tensor> = Vec::new();

        for batch in data_loader {
            let (x, _) = batch;
            let token_embeddings = token_layer.forward(&x)?;
            
            let len = x.dim(1)? as i64;
            let positions = Tensor::arange(0i64, len, &self.device)?.to_dtype(DType::I64)?.unsqueeze(0)?;
            let positions_embeddings = pos_layer.forward(&positions)?;
            
            let input_embeddings = token_embeddings.broadcast_add(&positions_embeddings)?;
            
            result.push(input_embeddings);
        }

        Ok(result)
    }
}