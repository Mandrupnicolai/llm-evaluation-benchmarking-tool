use anyhow::Context;
use chrono::Local;
use csv::Writer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use crate::dataset::DatasetItem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResult {
    pub id: String,
    pub prompt: String,
    pub reference: String,
    pub output: String,
    pub scores: HashMap<String, f64>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagBreakdown {
    pub tag: String,
    pub count: usize,
    pub metrics: HashMap<String, f64>,
}

impl ItemResult {
    pub fn from_parts(
        item: &DatasetItem,
        output: String,
        scores: HashMap<String, f64>,
    ) -> Self {
        Self {
            id: item.id.clone(),
            prompt: item.prompt.clone(),
            reference: item.reference.clone(),
            output,
            scores,
            tags: item.tags.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub run_id: String,
    pub model_name: String,
    pub metrics: Vec<String>,
    pub summary: HashMap<String, f64>,
    pub tag_summary: HashMap<String, HashMap<String, f64>>,
    pub tag_breakdown: Vec<TagBreakdown>,
    pub items: Vec<ItemResult>,
}

impl Report {
    pub fn from_results(
        run_id: String,
        model_name: String,
        metrics: Vec<String>,
        items: Vec<ItemResult>,
    ) -> Self {
        let summary = compute_summary(&metrics, &items);
        let tag_summary = compute_tag_summary(&metrics, &items);
        let tag_breakdown = compute_tag_breakdown(&metrics, &items);
        Self {
            run_id,
            model_name,
            metrics,
            summary,
            tag_summary,
            tag_breakdown,
            items,
        }
    }
}

pub fn new_run_id() -> String {
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}

pub fn write_predictions(path: &str, items: &[ItemResult]) -> anyhow::Result<()> {
    let mut lines = String::new();
    for item in items {
        let json = serde_json::to_string(item)?;
        lines.push_str(&json);
        lines.push('\n');
    }
    std::fs::write(path, lines)?;
    Ok(())
}

pub fn write_report(path: &str, report: &Report) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn write_csv_summary(path: &str, report: &Report) -> anyhow::Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["scope", "metric", "value"])?;
    for metric in &report.metrics {
        let value = report.summary.get(metric).cloned().unwrap_or(0.0);
        writer.write_record(["overall", metric, &format!("{:.4}", value)])?;
    }

    for (tag, metrics) in &report.tag_summary {
        for metric in &report.metrics {
            let value = metrics.get(metric).cloned().unwrap_or(0.0);
            writer.write_record([&format!("tag:{}", tag), metric, &format!("{:.4}", value)])?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub fn write_html_report(path: &str, report: &Report) -> anyhow::Result<()> {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\">");
    html.push_str("<title>LLM Evaluation Report</title>");
    html.push_str("<style>");
    html.push_str("body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111;}");
    html.push_str("table{border-collapse:collapse;width:100%;margin:12px 0;}");
    html.push_str("th,td{border:1px solid #ddd;padding:8px;text-align:left;}");
    html.push_str("th{background:#f3f3f3;}");
    html.push_str(".meta{color:#555;font-size:14px;}");
    html.push_str(".tag{margin-top:24px;}");
    html.push_str(".filter{margin:12px 0;padding:8px 10px;width:320px;max-width:100%;}");
    html.push_str(".chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;}");
    html.push_str(".chip{padding:6px 14px;border-radius:999px;background:#eef1ff;cursor:pointer;display:inline-block;}");
    html.push_str(".chip.active{background:#2c7dff;color:#fff;}");
    html.push_str("</style></head><body>");
    html.push_str("<h1>LLM Evaluation Report</h1>");
    html.push_str(&format!(
        "<p class=\"meta\">Run ID: {} | Model: {}</p>",
        report.run_id, report.model_name
    ));

    html.push_str("<h2>Summary</h2><table><tr><th>Metric</th><th>Value</th></tr>");
    for metric in &report.metrics {
        let value = report.summary.get(metric).cloned().unwrap_or(0.0);
        html.push_str(&format!(
            "<tr><td>{}</td><td>{:.4}</td></tr>",
            escape_html(metric),
            value
        ));
    }
    html.push_str("</table>");

    if !report.tag_summary.is_empty() {
        html.push_str("<h2>Per-Tag Summary</h2>");
        for (tag, metrics) in &report.tag_summary {
            html.push_str(&format!(
                "<div class=\"tag\"><h3>Tag: {}</h3>",
                escape_html(tag)
            ));
            html.push_str("<table><tr><th>Metric</th><th>Value</th></tr>");
            for metric in &report.metrics {
                let value = metrics.get(metric).cloned().unwrap_or(0.0);
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{:.4}</td></tr>",
                    escape_html(metric),
                    value
                ));
            }
            html.push_str("</table></div>");
        }
    }

    html.push_str("<h2>Items</h2>");
    html.push_str("<div class=\"chips\" id=\"tagChips\"></div>");
    html.push_str("<input class=\"filter\" id=\"tagFilter\" placeholder=\"Filter by tag (e.g. math)\" />");
    html.push_str("<table id=\"itemsTable\">");
    html.push_str("<tr><th>ID</th><th>Tags</th><th>Prompt</th><th>Reference</th><th>Output</th><th>Scores</th></tr>");
    for item in &report.items {
        let mut score_fragments: Vec<String> = Vec::new();
        for metric in &report.metrics {
            if let Some(value) = item.scores.get(metric) {
                score_fragments.push(format!("{}: {:.3}", metric, value));
            }
        }
        let tags = if item.tags.is_empty() {
            "".to_string()
        } else {
            item.tags.join(", ")
        };
        html.push_str(&format!(
            "<tr data-tags=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&tags),
            escape_html(&item.id),
            escape_html(&tags),
            escape_html(&item.prompt),
            escape_html(&item.reference),
            escape_html(&item.output),
            escape_html(&score_fragments.join(", "))
        ));
    }
    html.push_str("</table>");
    html.push_str("<script>");
    html.push_str("const input=document.getElementById('tagFilter');");
    html.push_str("const rows=[...document.querySelectorAll('#itemsTable tr[data-tags]')];");
    html.push_str("const chipWrap=document.getElementById('tagChips');");
    html.push_str("const tags=[...new Set(rows.flatMap(r=>(r.dataset.tags||'').split(',').map(t=>t.trim()).filter(Boolean)))]).sort();");
    html.push_str("function renderChips(){chipWrap.innerHTML='';");
    html.push_str("const all=document.createElement('span');all.className='chip active';all.textContent='All';");
    html.push_str("all.onclick=()=>{setActive(all);input.value='';applyFilter();};chipWrap.appendChild(all);");
    html.push_str("tags.forEach(tag=>{const chip=document.createElement('span');chip.className='chip';chip.textContent=tag;");
    html.push_str("chip.onclick=()=>{setActive(chip);input.value=tag;applyFilter();};chipWrap.appendChild(chip);});}");
    html.push_str("function setActive(active){[...chipWrap.children].forEach(c=>c.classList.remove('active'));active.classList.add('active');}");
    html.push_str("function applyFilter(){const q=input.value.toLowerCase().trim();rows.forEach(r=>{const tags=(r.dataset.tags||'').toLowerCase();");
    html.push_str("r.style.display=!q||tags.includes(q)?'':'none';});}");
    html.push_str("input.addEventListener('input',()=>{applyFilter();});");
    html.push_str("renderChips();");
    html.push_str("</script>");
    html.push_str("</body></html>");

    std::fs::write(path, html)?;
    Ok(())
}

pub fn read_predictions_map(path: &str) -> anyhow::Result<HashMap<String, String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open predictions: {}", path))?;
    let reader = BufReader::new(file);

    let mut map = HashMap::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("Invalid JSONL at line {}", line_no + 1))?;
        let id = value.get("id").and_then(|v| v.as_str());
        let output = value.get("output").and_then(|v| v.as_str());
        match (id, output) {
            (Some(id), Some(output)) => {
                map.insert(id.to_string(), output.to_string());
            }
            _ => continue,
        }
    }

    Ok(map)
}

fn compute_summary(metrics: &[String], items: &[ItemResult]) -> HashMap<String, f64> {
    let mut summary = HashMap::new();
    for metric in metrics {
        let mut total = 0.0;
        let mut count = 0.0;
        for item in items {
            if let Some(score) = item.scores.get(metric) {
                total += *score;
                count += 1.0;
            }
        }
        let avg = if count == 0.0 { 0.0 } else { total / count };
        summary.insert(metric.clone(), avg);
    }
    summary
}

fn compute_tag_summary(
    metrics: &[String],
    items: &[ItemResult],
) -> HashMap<String, HashMap<String, f64>> {
    let mut tag_items: HashMap<String, Vec<&ItemResult>> = HashMap::new();
    for item in items {
        for tag in &item.tags {
            tag_items.entry(tag.clone()).or_default().push(item);
        }
    }

    let mut summary = HashMap::new();
    for (tag, items) in tag_items {
        let mut metrics_map = HashMap::new();
        for metric in metrics {
            let mut total = 0.0;
            let mut count = 0.0;
            for item in &items {
                if let Some(score) = item.scores.get(metric) {
                    total += *score;
                    count += 1.0;
                }
            }
            let avg = if count == 0.0 { 0.0 } else { total / count };
            metrics_map.insert(metric.clone(), avg);
        }
        summary.insert(tag, metrics_map);
    }

    summary
}

fn compute_tag_breakdown(metrics: &[String], items: &[ItemResult]) -> Vec<TagBreakdown> {
    let mut tag_items: HashMap<String, Vec<&ItemResult>> = HashMap::new();
    for item in items {
        for tag in &item.tags {
            tag_items.entry(tag.clone()).or_default().push(item);
        }
    }

    let mut breakdown: Vec<TagBreakdown> = Vec::new();
    for (tag, items) in tag_items {
        let mut metrics_map = HashMap::new();
        for metric in metrics {
            let mut total = 0.0;
            let mut count = 0.0;
            for item in &items {
                if let Some(score) = item.scores.get(metric) {
                    total += *score;
                    count += 1.0;
                }
            }
            let avg = if count == 0.0 { 0.0 } else { total / count };
            metrics_map.insert(metric.clone(), avg);
        }

        breakdown.push(TagBreakdown {
            tag,
            count: items.len(),
            metrics: metrics_map,
        });
    }

    breakdown.sort_by(|a, b| a.tag.cmp(&b.tag));
    breakdown
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
