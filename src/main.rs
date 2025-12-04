mod encoder;
mod dataloader;

use std::fs;

use candle_core::{Device, Error, Tensor, DType};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use tiktoken_rs::o200k_base;
use dataloader::{DataSet, DataLoader};

fn main() -> Result<(), Error> {
    let encoder = o200k_base().unwrap();
    let data = fs::read_to_string("test-files/the-verdict.txt")?;
    let data_set = DataSet::new(data.as_str(), encoder, 4)?;
    
    // Configuration
    let vocab_size = data_set.vocab_size.try_into().unwrap();
    let output_dim = 256;
    let context_length = 1024;
    let batch_size = 8;
    let device = Device::Cpu;
    let dtype = DType::F32;

    let vb = VarBuilder::zeros(dtype.clone(), &device);

    let data_loader = DataLoader::new(data_set, batch_size, true, true, device.clone());
    
    let token_embedding_layer = embedding(vocab_size, output_dim, vb.pp("token_embeddings"))?;
    let pos_embedding_layer = embedding(context_length, output_dim, vb.pp("pos_embeddings"))?;
    
    for batch in data_loader {
        let (x, _) = batch;
        let token_embeddings = token_embedding_layer.forward(&x)?;
        
        let len = x.dim(1)? as i64;
        let positions = Tensor::arange(0i64, len, &device)?.to_dtype(DType::I64)?.unsqueeze(0)?;
        let positions_embeddings = pos_embedding_layer.forward(&positions)?;
        
        let input_embeddings = token_embeddings.broadcast_add(&positions_embeddings)?;
        
        println!("{:?}", input_embeddings);
    }

    Ok(())
}
