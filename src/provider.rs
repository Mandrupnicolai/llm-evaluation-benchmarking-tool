use anyhow::{bail, Context, Result};
use reqwest::blocking::Client;
use serde_json::json;

use crate::config::ModelConfig;

pub trait ModelProvider {
    fn name(&self) -> &str;
    fn generate(&self, prompt: &str) -> Result<String>;
}

pub fn build_provider(config: &ModelConfig) -> Result<Box<dyn ModelProvider>> {
    match config.provider_type.as_str() {
        "mock" => Ok(Box::new(MockProvider::new(config.name.clone()))),
        "openai" => Ok(Box::new(OpenAIProvider::from_config(config)?)),
        other => bail!("Unsupported provider type: {}", other),
    }
}

pub struct MockProvider {
    name: String,
}

impl MockProvider {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl ModelProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(&self, prompt: &str) -> Result<String> {
        if prompt.to_lowercase().contains("2 + 2") || prompt.to_lowercase().contains("2+2") {
            return Ok("4".to_string());
        }
        Ok(format!("Answer: {}", prompt))
    }
}

pub struct OpenAIProvider {
    name: String,
    api_key: String,
    base_url: String,
    instructions: Option<String>,
    temperature: f32,
    client: Client,
}

impl OpenAIProvider {
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let api_key = resolve_api_key(config.api_key.as_deref())?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Ok(Self {
            name: config.name.clone(),
            api_key,
            base_url,
            instructions: config.instructions.clone(),
            temperature: config.temperature,
            client: Client::new(),
        })
    }

    fn build_body(&self, prompt: &str) -> serde_json::Value {
        let mut body = json!({
            "model": self.name,
            "input": prompt,
            "temperature": self.temperature,
        });

        if let Some(instructions) = &self.instructions {
            body["instructions"] = json!(instructions);
        }

        body
    }

    fn parse_output(&self, value: &serde_json::Value) -> Result<String> {
        if let Some(text) = value.get("output_text").and_then(|v| v.as_str()) {
            return Ok(text.to_string());
        }

        if let Some(outputs) = value.get("output").and_then(|v| v.as_array()) {
            let mut chunks: Vec<String> = Vec::new();
            for output in outputs {
                if output.get("type").and_then(|v| v.as_str()) != Some("message") {
                    continue;
                }
                if let Some(content) = output.get("content").and_then(|v| v.as_array()) {
                    for part in content {
                        if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                chunks.push(text.to_string());
                            }
                        }
                    }
                }
            }
            if !chunks.is_empty() {
                return Ok(chunks.join(""));
            }
        }

        bail!("OpenAI response did not contain output text")
    }
}

impl ModelProvider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&self.build_body(prompt))
            .send()
            .context("OpenAI request failed")?;

        let status = response.status();
        let value: serde_json::Value = response.json().context("Invalid JSON response")?;
        if !status.is_success() {
            bail!(
                "OpenAI API error ({}): {}",
                status,
                value
                    .get("error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error")
            );
        }

        self.parse_output(&value)
    }
}

fn resolve_api_key(value: Option<&str>) -> Result<String> {
    if let Some(raw) = value {
        if let Some(stripped) = raw.strip_prefix("env:") {
            return std::env::var(stripped)
                .with_context(|| format!("Missing environment variable: {}", stripped));
        }
        return Ok(raw.to_string());
    }

    std::env::var("OPENAI_API_KEY").context("Missing OPENAI_API_KEY")
}
