use anyhow::{Context, Result};
use serde::Deserialize;
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::config::EvalConfig;
use crate::dataset::{parse_dataset_str, DatasetItem};
use crate::metrics::compute_metrics;
use crate::provider::build_provider;
use crate::report::{ItemResult, Report};

const INDEX_HTML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/ui/index.html"));
const APP_JS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/ui/app.js"));
const STYLE_CSS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/ui/style.css"));

#[derive(Debug, Deserialize)]
struct RunRequest {
    config_toml: String,
    dataset_jsonl: String,
}

pub fn serve_ui(host: &str, port: u16) -> Result<()> {
    let address = format!("{}:{}", host, port);
    let server = Server::http(&address)
        .map_err(|err| anyhow::anyhow!("Failed to bind {}: {}", address, err))?;
    println!("Serving UI at http://{}/", address);

    for request in server.incoming_requests() {
        let method = request.method();
        let url = request.url().to_string();
        if method == &Method::Get && (url == "/" || url == "/index.html") {
            let response = Response::from_string(INDEX_HTML)
                .with_header(header("Content-Type", "text/html; charset=utf-8"));
            let _ = request.respond(response);
        } else if method == &Method::Get && url == "/app.js" {
            let response = Response::from_string(APP_JS)
                .with_header(header("Content-Type", "application/javascript; charset=utf-8"));
            let _ = request.respond(response);
        } else if method == &Method::Get && url == "/style.css" {
            let response = Response::from_string(STYLE_CSS)
                .with_header(header("Content-Type", "text/css; charset=utf-8"));
            let _ = request.respond(response);
        } else if method == &Method::Post && url == "/api/run" {
            let result = handle_run_request(request);
            if let Err(err) = result {
                eprintln!("Run request error: {}", err);
            }
        } else {
            let response = Response::from_string("Not Found")
                .with_status_code(StatusCode(404));
            let _ = request.respond(response);
        }
    }

    Ok(())
}

fn handle_run_request(mut request: tiny_http::Request) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;
    let payload: RunRequest = serde_json::from_str(&body).context("Invalid JSON payload")?;

    let cfg: EvalConfig = toml::from_str(&payload.config_toml)
        .context("Invalid TOML config")?;
    let dataset = parse_dataset_str(&payload.dataset_jsonl)
        .context("Invalid dataset JSONL")?;

    let items = apply_max_items(dataset, cfg.run.max_items);
    let provider = build_provider(&cfg.model)?;

    let mut results: Vec<ItemResult> = Vec::new();
    for item in items.iter() {
        let output = provider.generate(&item.prompt)?;
        let scores = compute_metrics(&cfg.metrics.items, &output, &item.reference);
        results.push(ItemResult::from_parts(item, output, scores));
    }

    let report = Report::from_results(
        crate::report::new_run_id(),
        provider.name().to_string(),
        cfg.metrics.items.clone(),
        results,
    );

    let json = serde_json::to_string_pretty(&report)?;
    let response = Response::from_string(json)
        .with_header(header("Content-Type", "application/json; charset=utf-8"));
    request.respond(response)?;

    Ok(())
}

fn apply_max_items(items: Vec<DatasetItem>, max_items: usize) -> Vec<DatasetItem> {
    if max_items == 0 || items.len() <= max_items {
        items
    } else {
        items.into_iter().take(max_items).collect()
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name, value).expect("valid header")
}
