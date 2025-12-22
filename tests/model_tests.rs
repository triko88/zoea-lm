use burn::{
    backend::NdArray,
    module::Module,
    tensor::{Distribution, ElementConversion, Int, Tensor},
};
use zoea_lm::{
    gpt_config::GPTConfig,
    gpt_model::GPTModel,
    transformer::{TransformerBlock, TransformerConfig},
};

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

    let block = TransformerBlock::<TestBackend>::new(config, &device);

    let input: Tensor<TestBackend, 3> =
        Tensor::random([1, 8, 64], Distribution::Normal(0.0, 1.0), &device);

    let output = block.forward(input.clone());
    assert_eq!(output.dims(), input.dims());

    let residual: f32 = (output - input).abs().mean().into_scalar().elem();
    assert!(residual > 0.0, "Output should differ from input");
}

#[test]
fn test_gpt_model_forward_pass() {
    let device = Default::default();
    let config = GPTConfig::tiny();

    let (batch_size, token_num, vocab_size) = (12, 8, config.vocab_size);
    let model = GPTModel::<TestBackend>::new(config, &device);

    let input: Tensor<TestBackend, 2, Int> = Tensor::random(
        [batch_size, token_num],
        Distribution::Uniform(0.0, vocab_size as f64),
        &device,
    );

    let output = model.forward(input);

    assert_eq!(output.dims(), [batch_size, token_num, vocab_size])
}

#[test]
fn test_gpt_model_single_token() {
    let device = Default::default();
    let config = GPTConfig::tiny();
    let vocab_size = config.vocab_size;

    let model = GPTModel::<TestBackend>::new(config, &device);

    let input: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[42]], &device);

    let output = model.forward(input);

    assert_eq!(output.dims(), [1, 1, vocab_size]);
}
