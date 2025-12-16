mod attention;
mod dataloader;
mod encoder;

use std::collections::HashMap;
use std::fs;

use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use tiktoken_rs::o200k_base;

use crate::attention::SelfAttention;
use crate::dataloader::{DataLoader, DataSet};
use crate::encoder::Encoder;

fn load_weights(
    input_dim: usize,
    output_dim: usize,
    context_length: usize,
    device: Device,
) -> Result<HashMap<String, Tensor>, Error> {
    let mut weight_map: HashMap<String, Tensor> = HashMap::new();
    weight_map.insert(
        "token_embeddings.weight".to_string(),
        Tensor::randn(0.0, 0.2, (input_dim, output_dim), &device)?,
    );
    weight_map.insert(
        "pos_embeddings.weight".to_string(),
        Tensor::randn(0.0, 0.2, (context_length, output_dim), &device)?,
    );
    weight_map.insert(
        "queries.weight".to_string(),
        Tensor::randn(0.0, 0.2, (output_dim, output_dim), &device)?,
    );
    weight_map.insert(
        "queries.bias".to_string(),
        Tensor::randn(0.0, 0.0, (output_dim,), &device)?,
    );
    weight_map.insert(
        "keys.weight".to_string(),
        Tensor::randn(0.0, 0.2, (output_dim, output_dim), &device)?,
    );
    weight_map.insert(
        "keys.bias".to_string(),
        Tensor::randn(0.0, 0.0, (output_dim,), &device)?,
    );
    weight_map.insert(
        "values.weight".to_string(),
        Tensor::randn(0.0, 0.2, (output_dim, output_dim), &device)?,
    );
    weight_map.insert(
        "values.bias".to_string(),
        Tensor::randn(0.0, 0.0, (output_dim,), &device)?,
    );

    Ok(weight_map)
}

fn main() -> Result<(), Error> {
    let encoder = o200k_base().unwrap();
    let data = fs::read_to_string("test-files/small-text-sample.txt")?;
    let data_set = DataSet::new(data.as_str(), encoder, 4)?;

    // Configuration
    let vocab_size = data_set.vocab_size.try_into().unwrap();
    let output_dim = 256;
    let context_length = 1024;
    let batch_size = 8;
    let head_count = 8;

    let weight_map = load_weights(vocab_size, output_dim, context_length, Device::Cpu)?;

    let builder = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);

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
        head_count,
        &builder,
    )?;
    let context_vector = self_attn.forward(&input_embeddings, true)?;

    println!("{:?}", context_vector.to_vec3::<f32>()?);

    Ok(())
}
