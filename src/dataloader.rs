use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::{self, Dataset, InMemDataset},
    },
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use std::sync::Arc;
use tiktoken_rs::CoreBPE;

#[derive(Clone, Debug)]
pub struct DataItem {
    pub current_window: Vec<u32>,
    pub target_window: Vec<u32>,
}

pub struct DataSet {
    pub data: InMemDataset<DataItem>,
}

impl DataSet {
    pub fn new(input: &str, encoder: CoreBPE, window_size: usize, stride: usize) -> Self {
        let raw_tokens = encoder.encode_with_special_tokens(input);

        let mut current_window = raw_tokens
            .windows(window_size)
            .step_by(stride)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<Vec<u32>>>();

        let target_window = current_window[1..].to_vec();

        let _ = current_window.pop();

        let data = current_window
            .into_iter()
            .zip(target_window.into_iter())
            .map(|(current_window, target_window)| DataItem {
                current_window,
                target_window,
            })
            .collect::<Vec<_>>();

        Self {
            data: InMemDataset::new(data),
        }
    }
}

impl Dataset<DataItem> for DataSet {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, idx: usize) -> Option<DataItem> {
        self.data.get(idx)
    }
}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<B, DataItem, DataBatch<B>> for DataBatcher<B> {
    fn batch(&self, batch: Vec<DataItem>, device: &B::Device) -> DataBatch<B> {
        let batch_size = batch.len();
        let seq_len = batch[0].current_window.len();

        let device: &B::Device = device;

        let mut input_data = Vec::with_capacity(batch_size * seq_len);
        let mut target_data = Vec::with_capacity(batch_size * seq_len);

        for item in batch {
            input_data.extend_from_slice(&item.current_window);
            target_data.extend_from_slice(&item.target_window);
        }

        let shape = [batch_size, seq_len];

        let inputs = Tensor::from_data(TensorData::new(input_data, shape), device);
        let targets = Tensor::from_data(TensorData::new(target_data, shape), device);

        DataBatch { inputs, targets }
    }
}

pub fn create_dataloader<B: Backend>(
    text: &str,
    encoder: CoreBPE,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    num_workers: usize,
) -> Arc<dyn DataLoader<B, DataBatch<B>>> {
    let dataset = DataSet::new(text, encoder, max_length, stride);
    let dataset = Arc::new(dataset.data);

    DataLoaderBuilder::new(DataBatcher {
        device: B::Device::default(),
    })
    .batch_size(batch_size)
    .num_workers(num_workers)
    .build(dataset)
}
