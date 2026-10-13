//! Descriptions of models available through the static dispatcher.
use ferrots_core::model::{ModelDescriptor, ParameterSchema};

const DEVICES: &[&str] = if cfg!(feature = "cuda") {
    &["cpu", "cuda"]
} else {
    &["cpu"]
};
const LOCAL: ModelDescriptor = ModelDescriptor {
    name: "last_value",
    version: 1,
    source_repository: None,
    source_sha: None,
    source_files: &["crates/ferrots-nn/src/baselines.rs"],
    parameters: &[],
    task: "forecast",
    target_mode: "input_subset",
    history_time: false,
    future_known: false,
    missing_input_mask: false,
    supervision_mask: true,
    precisions: &["f32"],
    build_devices: DEVICES,
    reproduction_status: "unit_verified",
};
pub static MODELS: &[ModelDescriptor] = &[
    ModelDescriptor {
        name: "informer",
        source_repository: Some("https://github.com/thuml/Time-Series-Library"),
        source_sha: Some(crate::informer::UPSTREAM_SHA),
        source_files: &[
            "models/Informer.py",
            "layers/Embed.py",
            "layers/Transformer_EncDec.py",
            "layers/SelfAttention_Family.py",
        ],
        history_time: true,
        reproduction_status: "preview",
        parameters: &[
            ParameterSchema {
                name: "d_model",
                value_type: "positive_even_integer",
                default: "512",
            },
            ParameterSchema {
                name: "n_heads",
                value_type: "positive_integer_dividing_d_model",
                default: "8",
            },
            ParameterSchema {
                name: "d_ff",
                value_type: "positive_integer",
                default: "2048",
            },
            ParameterSchema {
                name: "e_layers",
                value_type: "positive_integer",
                default: "2",
            },
            ParameterSchema {
                name: "d_layers",
                value_type: "positive_integer",
                default: "1",
            },
            ParameterSchema {
                name: "factor",
                value_type: "positive_integer",
                default: "3",
            },
            ParameterSchema {
                name: "distil",
                value_type: "boolean",
                default: "true",
            },
            ParameterSchema {
                name: "label_len",
                value_type: "positive_integer_at_most_lookback",
                default: "48",
            },
            ParameterSchema {
                name: "dropout",
                value_type: "finite_probability_below_one",
                default: "0.1",
            },
            ParameterSchema {
                name: "activation",
                value_type: "gelu_or_relu",
                default: "gelu",
            },
        ],
        ..LOCAL
    },
    ModelDescriptor {
        name: "itransformer",
        source_repository: Some("https://github.com/thuml/iTransformer"),
        source_sha: Some(crate::itransformer::UPSTREAM_SHA),
        source_files: &[
            "model/iTransformer.py",
            "layers/Embed.py",
            "layers/Transformer_EncDec.py",
            "layers/SelfAttention_Family.py",
        ],
        history_time: true,
        reproduction_status: "p2_cuda_formal_pass",
        parameters: &[
            ParameterSchema {
                name: "d_model",
                value_type: "positive_integer",
                default: "512",
            },
            ParameterSchema {
                name: "n_heads",
                value_type: "positive_integer_dividing_d_model",
                default: "8",
            },
            ParameterSchema {
                name: "d_ff",
                value_type: "positive_integer",
                default: "2048",
            },
            ParameterSchema {
                name: "e_layers",
                value_type: "positive_integer",
                default: "2",
            },
            ParameterSchema {
                name: "dropout",
                value_type: "finite_probability_below_one",
                default: "0.1",
            },
            ParameterSchema {
                name: "activation",
                value_type: "gelu_or_relu",
                default: "gelu",
            },
            ParameterSchema {
                name: "use_norm",
                value_type: "boolean",
                default: "true",
            },
        ],
        ..LOCAL
    },
    LOCAL,
    ModelDescriptor {
        name: "seasonal_naive",
        parameters: &[ParameterSchema {
            name: "period",
            value_type: "positive_integer",
            default: "required",
        }],
        ..LOCAL
    },
    ModelDescriptor {
        name: "linear",
        parameters: &[ParameterSchema {
            name: "individual",
            value_type: "boolean",
            default: "false",
        }],
        ..LOCAL
    },
    ModelDescriptor {
        name: "dlinear",
        source_repository: Some("https://github.com/thuml/Time-Series-Library"),
        source_sha: Some(crate::baselines::UPSTREAM_SHA),
        source_files: &["models/DLinear.py"],
        parameters: &[
            ParameterSchema {
                name: "individual",
                value_type: "boolean",
                default: "false",
            },
            ParameterSchema {
                name: "moving_avg",
                value_type: "positive_odd_integer",
                default: "25",
            },
        ],
        reproduction_status: "p1_cpu_cuda_formal_pass",
        ..LOCAL
    },
];
pub fn find(name: &str) -> Option<&'static ModelDescriptor> {
    MODELS.iter().find(|model| model.name == name)
}
