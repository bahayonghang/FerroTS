//! Model descriptions contain no tensor or backend dependencies.
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ParameterSchema {
    pub name: &'static str,
    pub value_type: &'static str,
    pub default: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModelDescriptor {
    pub name: &'static str,
    pub version: u32,
    pub source_repository: Option<&'static str>,
    pub source_sha: Option<&'static str>,
    pub source_files: &'static [&'static str],
    pub parameters: &'static [ParameterSchema],
    pub task: &'static str,
    pub target_mode: &'static str,
    pub history_time: bool,
    pub future_known: bool,
    pub missing_input_mask: bool,
    pub supervision_mask: bool,
    pub precisions: &'static [&'static str],
    pub build_devices: &'static [&'static str],
    pub reproduction_status: &'static str,
}
