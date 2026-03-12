use clap::{Parser, Subcommand};

mod config;
mod dataset;
mod metrics;
mod provider;
mod report;
mod server;

use dataset::{read_dataset, write_sample_dataset, DatasetItem};
use metrics::compute_metrics;
use provider::build_provider;
use report::{write_report, ItemResult, Report};

#[derive(Parser)]
#[command(name = "llm-evaluation-benchmarking-tool")]
#[command(about = "A small, extensible LLM evaluation benchmarking tool.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create sample config and dataset files.
    Init {
        /// Directory to write sample files into (default: current directory)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Run an evaluation using a config TOML file.
    Run {
        /// Path to config TOML file
        #[arg(long, default_value = "eval_config.toml")]
        config: String,
    },
    /// Score a predictions file against a dataset.
    Score {
        /// Path to dataset JSONL file
        #[arg(long)]
        dataset: String,
        /// Path to predictions JSONL file
        #[arg(long)]
        predictions: String,
        /// Metrics to compute (repeatable). If omitted, uses config defaults.
        #[arg(long)]
        metrics: Vec<String>,
        /// Output path for report JSON (default: <predictions>.report.json)
        #[arg(long)]
        output: Option<String>,
    },
    /// Serve a local web UI for running evaluations.
    Serve {
        /// Host to bind (default: 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind (default: 8080)
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { dir } => {
            let base_dir = dir.unwrap_or_else(|| ".".to_string());
            config::write_sample_config(&format!("{}/eval_config.toml", base_dir))?;
            write_sample_dataset(&format!("{}/data/samples.jsonl", base_dir))?;
            println!("Wrote sample config and dataset under {}", base_dir);
        }
        Commands::Run { config } => {
            let cfg = config::load_config(&config)?;
            let dataset = read_dataset(&cfg.dataset.path)?;
            let items = apply_max_items(dataset, cfg.run.max_items);

            let provider = build_provider(&cfg.model)?;
            let run_id = report::new_run_id();
            let run_dir = format!("{}/{}", cfg.output.dir, run_id);
            std::fs::create_dir_all(&run_dir)?;

            let mut results: Vec<ItemResult> = Vec::new();
            for item in items.iter() {
                let output = provider.generate(&item.prompt)?;
                let scores = compute_metrics(&cfg.metrics.items, &output, &item.reference);
                results.push(ItemResult::from_parts(item, output, scores));
            }

            let report = Report::from_results(
                run_id,
                provider.name().to_string(),
                cfg.metrics.items.clone(),
                results,
            );

            let predictions_path = format!("{}/predictions.jsonl", run_dir);
            let report_path = format!("{}/report.json", run_dir);

            report::write_predictions(&predictions_path, &report.items)?;
            write_report(&report_path, &report)?;
            report::write_csv_summary(&format!("{}/report.csv", run_dir), &report)?;
            report::write_html_report(&format!("{}/report.html", run_dir), &report)?;

            println!("Run complete. Report: {}", report_path);
        }
        Commands::Score {
            dataset,
            predictions,
            metrics,
            output,
        } => {
            let dataset_items = read_dataset(&dataset)?;
            let predictions_map = report::read_predictions_map(&predictions)?;
            let metric_list = if metrics.is_empty() {
                config::default_metrics()
            } else {
                metrics
            };

            let mut results: Vec<ItemResult> = Vec::new();
            for item in dataset_items.iter() {
                if let Some(output) = predictions_map.get(&item.id) {
                    let scores = compute_metrics(&metric_list, output, &item.reference);
                    results.push(ItemResult::from_parts(item, output.clone(), scores));
                }
            }

            let report = Report::from_results(
                report::new_run_id(),
                "external".to_string(),
                metric_list,
                results,
            );

            let report_path = output.unwrap_or_else(|| format!("{}.report.json", predictions));
            write_report(&report_path, &report)?;
            let csv_path = format!("{}.report.csv", predictions);
            let html_path = format!("{}.report.html", predictions);
            report::write_csv_summary(&csv_path, &report)?;
            report::write_html_report(&html_path, &report)?;

            println!("Scoring complete. Report: {}", report_path);
        }
        Commands::Serve { host, port } => {
            server::serve_ui(&host, port)?;
        }
    }

    Ok(())
}

fn apply_max_items(items: Vec<DatasetItem>, max_items: usize) -> Vec<DatasetItem> {
    if max_items == 0 || items.len() <= max_items {
        items
    } else {
        items.into_iter().take(max_items).collect()
    }
}
