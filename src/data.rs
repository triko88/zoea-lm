type TokenType = i64;
pub struct DataSet {
    pub input_ids: Vec<Vec<TokenType>>,
    pub target_ids: Vec<Vec<TokenType>>,
}

impl DataSet {
    pub fn new(token_ids: Vec<TokenType>, max_length: usize, stride: usize) -> Self {
        let input_ids: Vec<Vec<TokenType>> = token_ids
            .chunks(stride)
            .map(|chunk| {
                let mut chunk = chunk.to_vec();
                if chunk.len() < max_length {
                    chunk.extend(vec![0; max_length - chunk.len()]);
                }
                chunk
            })
            .collect();

        let target_ids: Vec<Vec<TokenType>> = input_ids[1..].to_vec();

        Self {
            input_ids,
            target_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn get(&self, index: usize) -> Option<(&Vec<TokenType>, &Vec<TokenType>)> {
        if index >= self.len() {
            return None;
        }

        Some((&self.input_ids[index], &self.target_ids[index]))
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty() && self.target_ids.is_empty()
    }
}
