//! TokenConv init against the PyTorch 2.14 constructor used by the pinned TSLib.
//! Expected prefixes are the first 16 values of
//! `runs/p3-development/20261014-d4-cuda/fixtures/init-torch-dropout0.1.safetensors`,
//! which is `torch.manual_seed(2021)` and Informer dropout 0.1 at the ETT shape.

use candle_core::Device;
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::informer::Informer;

fn prefix(weights: &candle_core::Tensor) -> Vec<f32> {
    weights.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..16].to_vec()
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

#[test]
#[allow(clippy::excessive_precision)]
fn token_conv_matches_torch_kaiming_normal_and_later_uniform_stays_aligned() {
    let spec = ModelSpec::Informer {
        d_model: 512,
        n_heads: 8,
        d_ff: 2048,
        e_layers: 2,
        d_layers: 1,
        factor: 3,
        distil: true,
        label_len: 48,
        dropout: 0.1,
        activation: Activation::Gelu,
    };
    let window = WindowSpec {
        lookback: 96,
        horizon: 720,
        gap: 0,
        stride: 1,
    };
    let model = Informer::new(&spec, &window, 7, 4, 2021, &Device::Cpu).unwrap();
    let weights = model.weights();
    let enc = prefix(&weights["enc_embedding.value_embedding.tokenConv.weight"]);
    let dec = prefix(&weights["dec_embedding.value_embedding.tokenConv.weight"]);
    let temporal = prefix(&weights["enc_embedding.temporal_embedding.embed.weight"]);
    let projection = prefix(&weights["decoder.projection.weight"]);
    let enc_expected = [
        -1.45033389e-01,
        -2.23664418e-01,
        1.91404119e-01,
        5.51241577e-01,
        7.20973909e-02,
        -5.31640470e-01,
        -1.40434965e-01,
        1.11434542e-01,
        4.46540833e-01,
        -3.95081580e-01,
        -3.21590126e-01,
        3.94045591e-01,
        -2.86915433e-02,
        5.70805848e-01,
        -8.12718987e-01,
        -3.63567799e-01,
    ];
    let dec_expected = [
        -5.81392758e-02,
        -4.68159206e-02,
        1.52236044e-01,
        2.65987158e-01,
        -2.81328022e-01,
        -1.53749600e-01,
        3.64623159e-01,
        2.91286968e-02,
        -3.38417917e-01,
        9.46699083e-03,
        -8.41092765e-02,
        7.06980526e-02,
        4.42616120e-02,
        -3.95408273e-01,
        3.88556242e-01,
        -3.40209574e-01,
    ];
    let temporal_expected = [
        -7.68623352e-02,
        2.94764698e-01,
        -1.18317246e-01,
        2.12412417e-01,
        1.31411076e-01,
        -2.05707192e-01,
        4.80368376e-01,
        -4.69193995e-01,
        -3.02253425e-01,
        4.33523595e-01,
        -4.00273740e-01,
        -3.17664623e-01,
        3.38827968e-02,
        -3.98492694e-01,
        -7.15301037e-02,
        4.45057034e-01,
    ];
    let projection_expected = [
        1.96982455e-02,
        -3.03033255e-02,
        1.22977179e-02,
        3.29858772e-02,
        1.53467469e-02,
        -2.96796001e-02,
        3.52216884e-02,
        -3.06834321e-04,
        -3.09700463e-02,
        3.77116501e-02,
        -4.35661487e-02,
        2.67147683e-02,
        6.61872001e-03,
        3.51902731e-02,
        1.75420605e-02,
        2.68130396e-02,
    ];
    let enc_gap = max_abs(&enc, &enc_expected);
    let dec_gap = max_abs(&dec, &dec_expected);
    assert!(
        enc_gap < 1e-5 && dec_gap < 1e-5,
        "tokenConv enc={enc_gap} dec={dec_gap}"
    );
    let temporal_gap = max_abs(&temporal, &temporal_expected);
    let projection_gap = max_abs(&projection, &projection_expected);
    assert!(
        temporal_gap < 1e-6 && projection_gap < 1e-6,
        "uniform stream temporal={temporal_gap} projection={projection_gap}"
    );
}
