use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetItem {
    pub id: String,
    pub prompt: String,
    pub reference: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

pub fn read_dataset(path: &str) -> anyhow::Result<Vec<DatasetItem>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open dataset: {}", path))?;
    let reader = BufReader::new(file);

    let mut items = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item: DatasetItem = serde_json::from_str(&line)
            .with_context(|| format!("Invalid JSONL at line {}", line_no + 1))?;
        items.push(item);
    }
    Ok(items)
}

pub fn parse_dataset_str(contents: &str) -> anyhow::Result<Vec<DatasetItem>> {
    let mut items = Vec::new();
    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: DatasetItem = serde_json::from_str(line)
            .with_context(|| format!("Invalid JSONL at line {}", line_no + 1))?;
        items.push(item);
    }
    Ok(items)
}

pub fn write_sample_dataset(path: &str) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let samples = vec![
        DatasetItem {
            id: "math-1".to_string(),
            prompt: "What is 2 + 2?".to_string(),
            reference: "4".to_string(),
            tags: vec!["math".to_string()],
        },
        DatasetItem {
            id: "fact-1".to_string(),
            prompt: "Name the capital of France.".to_string(),
            reference: "Paris".to_string(),
            tags: vec!["geography".to_string()],
        },
        DatasetItem {
            id: "summ-1".to_string(),
            prompt: "Summarize: LLMs are models trained on large datasets.".to_string(),
            reference: "LLMs are trained on large datasets.".to_string(),
            tags: vec!["summarization".to_string()],
        },
    ];

    let mut lines = String::new();
    for item in samples {
        let json = serde_json::to_string(&item)?;
        lines.push_str(&json);
        lines.push('\n');
    }

    std::fs::write(path, lines)?;
    Ok(())
}
