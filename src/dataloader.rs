use candle_core::{Device, Error, Tensor};
use rand::seq::SliceRandom;
use tiktoken_rs::CoreBPE;

#[derive(Debug)]
pub struct DataSet {
    pub current_window: Vec<Tensor>,
    pub target_window: Vec<Tensor>,
    pub vocab_size: u32,
}

impl DataSet {
    pub fn new(input: &str, encoder: CoreBPE, dimensions: usize) -> Result<Self, Error> {
        let tokens = encoder.encode_with_special_tokens(input);
        let token_windows = tokens.windows(dimensions).step_by(dimensions);
        
        let device = Device::Cpu;
        
        let mut current_window = token_windows.map(|window| Tensor::new(window, &device))
                                    .collect::<Result<Vec<Tensor>, Error>>()?;

        let target_window = current_window[1..].to_vec();
        let _ = current_window.pop();
        
        Ok(Self {
            current_window,
            target_window,
            vocab_size: tokens.iter().max().unwrap() + 1,
        })
    }
    
    pub fn len(&self) -> usize {
        self.current_window.len()
    }
    
    pub fn get(&self, idx: usize) -> Option<(&Tensor, &Tensor)> {
        if idx >= self.current_window.len() {
            None
        } else {
            Some((&self.current_window[idx], &self.target_window[idx]))
        }
    }
}

pub struct DataLoader {
    dataset:        DataSet,
    batch_size:     usize,
    shuffle:        bool,
    drop_last:      bool,
    indeces:       Vec<usize>,
    current_idx:    usize,
    device:         Device,
}

impl DataLoader {
    pub fn new(dataset: DataSet, batch_size: usize, shuffle: bool, drop_last: bool, device: Device) -> Self {
        let indeces: Vec<usize> = (0..dataset.len()).collect();
        
        let mut loader = Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indeces,
            current_idx: 0,
            device
        };
        
        if shuffle {
            loader.shuffle_indeces();
        }
        
        loader
    }
    
    fn shuffle_indeces(&mut self) {
        let mut rng = rand::rng();
        self.indeces.shuffle(&mut rng);
    }
    
    pub fn reset(&mut self) {
        self.current_idx = 0;

        if self.shuffle {
            self.shuffle_indeces();
        }
    }
    
    pub fn count_batches(&self) -> usize {
        let count = if self.drop_last {
            self.dataset.len()
        } else {
            self.dataset.len() + self.batch_size - 1
        };

        count / self.batch_size
    }
    
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }
        
        let remaining = self.dataset.len() - self.current_idx;
        let actual_batch_size = if remaining < self.batch_size {
            if self.drop_last {
                return None;
            }
            remaining
        } else {
            self.batch_size
        };
        
        let mut current_batch = Vec::with_capacity(actual_batch_size);
        let mut target_batch = Vec::with_capacity(actual_batch_size);

        for _ in 0..actual_batch_size {
            let idx = self.indeces[self.current_idx];
            let (current, target) = self.dataset.get(idx).unwrap();
            current_batch.push(current.clone());
            target_batch.push(target.clone());

            self.current_idx += 1;
        }
        
        let current_refs: Vec<&Tensor> = current_batch.iter().collect();
        let target_refs: Vec<&Tensor> = target_batch.iter().collect();
        
        let current_tensor = Tensor::stack(&current_refs, 0).unwrap();
        let target_tensor = Tensor::stack(&target_refs, 0).unwrap();
        
        Some((current_tensor, target_tensor))
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}