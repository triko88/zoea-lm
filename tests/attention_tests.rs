use burn::{
    backend::NdArray,
    tensor::{Distribution, ElementConversion, Tensor},
};
use zoea_lm::attention::{MultiHeadAttention, MultiHeadAttentionConfig};

type TestBackend = NdArray<f32>;

#[test]
fn test_attention_output_bounded() {
    let device = Default::default();

    let config = MultiHeadAttentionConfig {
        input_dimensions: 64,
        output_dimensions: 64,
        context_length: 32,
        num_heads: 4,
        drop_out: 0.0,
        with_bias: false,
    };

    let attention = MultiHeadAttention::<TestBackend>::new(config);

    let input: Tensor<TestBackend, 3> =
        Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

    let output = attention.forward(input).into_data();
    let values: &[f32] = output.as_slice().unwrap();

    for (x, &val) in values.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Value {} at index {} is not finite",
            val,
            x
        );
    }
}

#[test]
fn test_attention_head_configurations() {
    let device = Default::default();

    let configs = [
        (64, 1),  // Single head
        (64, 2),  // 2 heads
        (64, 4),  // 4 heads
        (64, 8),  // 8 heads
        (64, 64), // Head dim = 1
        (128, 8), // Larger model
    ];

    for (dimensions, num_heads) in configs {
        let config = MultiHeadAttentionConfig {
            input_dimensions: dimensions,
            output_dimensions: dimensions,
            context_length: 32,
            num_heads,
            drop_out: 0.0,
            with_bias: false,
        };

        let attention = MultiHeadAttention::<TestBackend>::new(config);

        let input: Tensor<TestBackend, 3> =
            Tensor::random([1, 8, dimensions], Distribution::Normal(0.0, 1.0), &device);

        let output = attention.forward(input);

        assert_eq!(
            output.shape().dims(),
            [1, 8, dimensions],
            "Failed for dimensions = {} and num_heads = {}",
            dimensions,
            num_heads
        );
    }
}

#[test]
fn test_attention_causality() {
    let device = Default::default();

    let config = MultiHeadAttentionConfig {
        input_dimensions: 32,
        output_dimensions: 32,
        context_length: 16,
        num_heads: 2,
        drop_out: 0.0,
        with_bias: false,
    };

    let attention = MultiHeadAttention::<TestBackend>::new(config);

    let input1: Tensor<TestBackend, 3> =
        Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);

    let mut input2_data = input1.clone().into_data();
    let slice = input2_data.as_mut_slice::<f32>().unwrap();

    for x in (7 * 32)..(8 * 32) {
        slice[x] = 999.0;
    }

    let input2 = Tensor::from_data(input2_data, &device);

    let output1 = attention.forward(input1);
    let output2 = attention.forward(input2);

    let out1_first7 = output1.clone().slice([0..1, 0..7, 0..32]);
    let out2_first7 = output2.clone().slice([0..1, 0..7, 0..32]);

    let diff: f32 = (out1_first7 - out2_first7)
        .abs()
        .mean()
        .into_scalar()
        .elem();

    assert!(
        diff < 1e-5,
        "First 7 tokens should be unaffected by changes to token 8, diff={}",
        diff
    );

    let out1_last7 = output1.slice([0..1, 7..8, 0..32]);
    let out2_last7 = output2.slice([0..1, 7..8, 0..32]);

    let diff: f32 = (out1_last7 - out2_last7).abs().mean().into_scalar().elem();

    assert!(
        diff > 0.1,
        "Last 7 tokens should be affected by changes to token 8, diff={}",
        diff
    );
}

#[test]
fn test_attention_with_bias() {
    let device = Default::default();

    let unbiased_config = MultiHeadAttentionConfig {
        input_dimensions: 64,
        output_dimensions: 64,
        context_length: 16,
        num_heads: 4,
        drop_out: 0.0,
        with_bias: false,
    };

    let biased_config = MultiHeadAttentionConfig {
        with_bias: true,
        ..unbiased_config
    };

    let unbiased_attention = MultiHeadAttention::<TestBackend>::new(unbiased_config);
    let biased_attention = MultiHeadAttention::<TestBackend>::new(biased_config);

    let dims = [2, 10, 64];

    let input: Tensor<TestBackend, 3> =
        Tensor::random(dims, Distribution::Normal(0.0, 1.0), &device);

    let unbiased_output = unbiased_attention.forward(input.clone());
    let biased_output = biased_attention.forward(input);

    assert_eq!(unbiased_output.dims(), dims);
    assert_eq!(biased_output.dims(), dims);
}

#[test]
fn test_attention_single_token() {
    let device = Default::default();

    let config = MultiHeadAttentionConfig {
        input_dimensions: 64,
        output_dimensions: 64,
        context_length: 16,
        num_heads: 4,
        drop_out: 0.0,
        with_bias: false,
    };

    let attention = MultiHeadAttention::<TestBackend>::new(config);

    let input: Tensor<TestBackend, 3> =
        Tensor::random([1, 1, 64], Distribution::Normal(0.0, 1.0), &device);

    let output = attention.forward(input);

    assert_eq!(output.dims(), [1, 1, 64]);
}
