use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Deserialize, Default, Clone)]
pub struct EvalConfig {
    #[serde(default)]
    pub dataset: DatasetConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub run: RunConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatasetConfig {
    pub path: String,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            path: "data/samples.jsonl".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    pub name: String,
    pub temperature: f32,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider_type: "mock".to_string(),
            name: "mock-echo".to_string(),
            temperature: 0.0,
            api_key: None,
            base_url: None,
            instructions: None,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    pub dir: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: "runs".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct MetricsConfig {
    pub items: Vec<String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            items: default_metrics(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RunConfig {
    pub max_items: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self { max_items: 0 }
    }
}

pub fn default_metrics() -> Vec<String> {
    vec![
        "exact_match".to_string(),
        "contains_ref".to_string(),
        "jaccard".to_string(),
        "bleu".to_string(),
        "bleu4".to_string(),
        "rouge_l".to_string(),
        "rouge_lsum".to_string(),
        "f1".to_string(),
    ]
}

pub fn load_config(path: &str) -> anyhow::Result<EvalConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path))?;
    let cfg: EvalConfig = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config: {}", path))?;
    Ok(cfg)
}

pub fn write_sample_config(path: &str) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let sample = r#"[dataset]
path = "data/samples.jsonl"

[model]
type = "mock"
name = "mock-echo"
temperature = 0.0
# For OpenAI, set:
# type = "openai"
# name = "gpt-4.1-mini"
# temperature = 0.2
# api_key = "env:OPENAI_API_KEY"
# base_url = "https://api.openai.com/v1"
# instructions = "Answer briefly."

[output]
dir = "runs"

[metrics]
items = ["exact_match", "contains_ref", "jaccard", "bleu", "bleu4", "rouge_l", "rouge_lsum", "f1"]

[run]
max_items = 0
"#;

    std::fs::write(path, sample)?;
    Ok(())
}
