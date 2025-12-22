use burn::{backend::WebGpu, data::dataloader::DataLoader};
use std::{env, fs};
use tiktoken_rs::o200k_base;

use zoea_lm::data;

fn main() {
    let filename = env::args().nth(1).expect("Filename not provided");

    let file_contents = fs::read_to_string(filename).expect("Failed to read file");
}
