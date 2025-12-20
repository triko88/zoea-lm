use burn::{
    backend::NdArray,
    tensor::{Distribution, ElementConversion, Tensor},
};
use zoea_lm::transformer::{TransformerBlock, TransformerConfig};

type TestBackend = NdArray<f32>;

#[test]
fn test_transformer_block_residual_connection() {
    let device = Default::default();

    let config = TransformerConfig {
        embedding_dims: 64,
        context_length: 32,
        num_heads: 4,
        drop_rate: 0.0,
        bias: false,
    };

    let block = TransformerBlock::<TestBackend>::new(config, device);

    let input: Tensor<TestBackend, 3> =
        Tensor::random([1, 8, 64], Distribution::Normal(0.0, 1.0), &device);

    let output = block.forward(input.clone());
    assert_eq!(output.dims(), input.dims());

    let residual: f32 = (output - input).abs().mean().into_scalar().elem();
    assert!(residual > 0.0, "Output should differ from input");
}
