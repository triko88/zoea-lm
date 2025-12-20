use burn::{backend::WebGpu, data::dataloader::DataLoader};
use std::{env, fs};
use tiktoken_rs::o200k_base;

use zoea_lm::dataloader;

fn main() {
    let filename = env::args().nth(1).expect("Filename not provided");

    let file_contents = fs::read_to_string(filename).expect("Failed to read file");

    let window_size = 5;
    let stride = 2;
    let batch_size = 1;
    let num_workers = 2;

    let loader = dataloader::create_dataloader::<WebGpu>(
        &file_contents,
        o200k_base().unwrap(),
        batch_size,
        window_size,
        stride,
        num_workers,
    );

    for batch in loader.iter().take(5) {
        println!("Batch: {:?}", batch);
    }
}
