mod attention;
mod dataloader;
mod encoder;

use std::fs;

use candle_core::{DType, Device, Error};
use candle_nn::VarBuilder;
use tiktoken_rs::o200k_base;

use crate::attention::SelfAttention;
use crate::dataloader::{DataLoader, DataSet};
use crate::encoder::Encoder;

fn main() -> Result<(), Error> {
    let encoder = o200k_base().unwrap();
    let data = fs::read_to_string("test-files/the-verdict.txt")?;
    let data_set = DataSet::new(data.as_str(), encoder, 4)?;

    // Configuration
    let vocab_size = data_set.vocab_size.try_into().unwrap();
    let output_dim = 256;
    let context_length = 1024;
    let batch_size = 8;
    let builder = VarBuilder::zeros(DType::F32, &Device::Cpu);

    let data_loader = DataLoader::new(data_set, batch_size, true, true, &builder);

    let local_encoder = Encoder::new(vocab_size, context_length, output_dim);

    let input_embeddings = local_encoder.get_input_embeddings(data_loader)?;

    println!("input embeddings: generated...");
    let attention_dimesion = input_embeddings.dim(2)?;
    let self_attn = SelfAttention::new(
        attention_dimesion,
        output_dim,
        context_length,
        0.1,
        &builder,
    )?;
    let context_vector = self_attn.forward(&input_embeddings, true)?;

    println!("{:?}", context_vector);

    Ok(())
}
