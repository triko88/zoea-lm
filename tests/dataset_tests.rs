use zoea_lm::data::DataSet;

#[test]
fn test_dataset_creation() {
    let tokens: Vec<i64> = (0..100).collect();
    let max_length = 10;
    let stride = 5;

    let dataset = DataSet::new(tokens, max_length, stride);

    assert!(!dataset.is_empty());
    assert!(dataset.len() > 0);
}

#[test]
fn test_dataset_input_target_alignment() {
    let tokens: Vec<i64> = (0..50).collect();
    let dataset = DataSet::new(tokens, 8, 4);

    for i in 0..dataset.len() {
        let (input, target) = dataset.get(i).unwrap();

        assert_eq!(input.len(), 8);
        assert_eq!(target.len(), 8);

        // Each target token should be input shifted by 1
        for j in 0..8 {
            assert_eq!(target[j], input[j] + 1);
        }
    }
}

#[test]
fn test_dataset_stride_behavior() {
    let tokens: Vec<i64> = (0..30).collect();
    let max_length = 10;
    let stride = 5;

    let dataset = DataSet::new(tokens, max_length, stride);

    // First sample starts at 0
    let (input0, _) = dataset.get(0).unwrap();
    assert_eq!(input0[0], 0);

    // Second sample starts at stride (5)
    let (input1, _) = dataset.get(1).unwrap();
    assert_eq!(input1[0], 5);
}
