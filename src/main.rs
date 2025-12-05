mod encoder;
mod dataloader;

use std::fs;

use candle_core::{Device, Error};
use tiktoken_rs::o200k_base;

use crate::dataloader::{DataSet, DataLoader};
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
    let device = Device::Cpu;

    let data_loader = DataLoader::new(data_set, batch_size, true, true, device.clone());
    
    let local_encoder = Encoder::new(vocab_size, context_length, output_dim, device);
    
    let input_layer = local_encoder.get_input_embeddings(data_loader)?;
    
    println!("{:?}", input_layer);

    Ok(())
}
