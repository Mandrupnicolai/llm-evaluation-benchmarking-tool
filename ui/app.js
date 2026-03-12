const datasetInput = document.getElementById('datasetInput');
const configInput = document.getElementById('configInput');
const runButton = document.getElementById('runButton');
const loadSample = document.getElementById('loadSample');
const exportJson = document.getElementById('exportJson');
const exportCsv = document.getElementById('exportCsv');
const exportHtml = document.getElementById('exportHtml');
const summaryEl = document.getElementById('summary');
const tagBreakdownEl = document.getElementById('tagBreakdown');
const itemsTable = document.getElementById('itemsTable').querySelector('tbody');
const statusEl = document.getElementById('runStatus');
const tagFilter = document.getElementById('tagFilter');
const tagChips = document.getElementById('tagChips');
const compareA = document.getElementById('compareA');
const compareB = document.getElementById('compareB');
const compareButton = document.getElementById('compareButton');
const compareOutput = document.getElementById('compareOutput');
const scaleToggle = document.getElementById('scaleToggle');

let currentReport = null;

const sampleDataset = `{"id":"math-1","prompt":"What is 2 + 2?","reference":"4","tags":["math"]}
{"id":"fact-1","prompt":"Name the capital of France.","reference":"Paris","tags":["geography"]}
{"id":"summ-1","prompt":"Summarize: LLMs are models trained on large datasets.","reference":"LLMs are trained on large datasets.","tags":["summarization"]}`;

const sampleConfig = `[dataset]
path = "data/samples.jsonl"

[model]
type = "mock"
name = "mock-echo"
temperature = 0.0

[output]
dir = "runs"

[metrics]
items = ["exact_match", "contains_ref", "jaccard", "bleu", "bleu4", "rouge_l", "rouge_lsum", "f1"]

[run]
max_items = 0
`;

loadSample.addEventListener('click', () => {
  datasetInput.value = sampleDataset;
  configInput.value = sampleConfig;
});

runButton.addEventListener('click', async () => {
  statusEl.textContent = 'Running...';
  statusEl.style.background = 'rgba(255, 122, 89, 0.12)';
  statusEl.style.color = '#e85b38';

  const payload = {
    dataset_jsonl: datasetInput.value,
    config_toml: configInput.value,
  };

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const report = await response.json();
    renderReport(report);
    currentReport = report;

    statusEl.textContent = 'Complete';
    statusEl.style.background = 'rgba(44, 125, 255, 0.1)';
    statusEl.style.color = '#2c7dff';
  } catch (err) {
    statusEl.textContent = 'Error';
    statusEl.style.background = 'rgba(255, 0, 0, 0.1)';
    statusEl.style.color = '#d13b3b';
    alert(`Run failed: ${err.message}`);
  }
});

exportJson.addEventListener('click', () => {
  if (!currentReport) return alert('Run an evaluation first.');
  downloadFile('report.json', JSON.stringify(currentReport, null, 2), 'application/json');
});

exportCsv.addEventListener('click', () => {
  if (!currentReport) return alert('Run an evaluation first.');
  downloadFile('report.csv', buildCsv(currentReport), 'text/csv');
});

exportHtml.addEventListener('click', () => {
  if (!currentReport) return alert('Run an evaluation first.');
  downloadFile('report.html', buildHtml(currentReport), 'text/html');
});

compareButton.addEventListener('click', () => {
  try {
    const reportA = JSON.parse(compareA.value);
    const reportB = JSON.parse(compareB.value);
    renderComparison(reportA, reportB, scaleToggle.checked);
  } catch (err) {
    alert('Invalid JSON in compare inputs.');
  }
});

function renderReport(report) {
  summaryEl.innerHTML = '';
  tagBreakdownEl.innerHTML = '';
  itemsTable.innerHTML = '';

  for (const metric of report.metrics) {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h3>${metric}</h3><strong>${(report.summary[metric] ?? 0).toFixed(4)}</strong>`;
    summaryEl.appendChild(card);
  }

  if (report.tag_breakdown && report.tag_breakdown.length) {
    for (const tagInfo of report.tag_breakdown) {
      const card = document.createElement('div');
      card.className = 'card';
      const lines = report.metrics.map((m) => `${m}: ${(tagInfo.metrics[m] ?? 0).toFixed(3)}`).join('<br/>');
      card.innerHTML = `<h3>${tagInfo.tag} (${tagInfo.count})</h3><strong>${lines}</strong>`;
      tagBreakdownEl.appendChild(card);
    }
  }

  const tags = new Set();
  report.items.forEach((item) => (item.tags || []).forEach((tag) => tags.add(tag)));
  renderTagChips([...tags]);

  for (const item of report.items) {
    const scores = report.metrics.map((m) => `${m}: ${(item.scores[m] ?? 0).toFixed(3)}`).join(', ');
    const row = document.createElement('tr');
    row.dataset.tags = (item.tags || []).join(',').toLowerCase();
    row.innerHTML = `
      <td>${escapeHtml(item.id)}</td>
      <td>${escapeHtml((item.tags || []).join(', '))}</td>
      <td>${escapeHtml(item.prompt)}</td>
      <td>${escapeHtml(item.reference)}</td>
      <td>${escapeHtml(item.output)}</td>
      <td>${escapeHtml(scores)}</td>
    `;
    itemsTable.appendChild(row);
  }
}

function renderComparison(reportA, reportB, fixedScale) {
  const metrics = Array.from(new Set([...(reportA.metrics || []), ...(reportB.metrics || [])]));
  const deltaScale = fixedScale ? 1.0 : computeDeltaScale(reportA, reportB, metrics);
  let html = '<h3>Summary Deltas (B - A)</h3>';
  html += '<table><tr><th>Metric</th><th>Run A</th><th>Run B</th><th>Delta</th><th>Spark</th><th>Tags</th></tr>';
  for (const metric of metrics) {
    const a = reportA.summary?.[metric] ?? 0;
    const b = reportB.summary?.[metric] ?? 0;
    const delta = b - a;
    const cls = delta >= 0 ? 'delta-positive' : 'delta-negative';
    const spark = buildTagSparkline(reportA, reportB, metric);
    html += `<tr><td>${metric}</td><td>${a.toFixed(4)}</td><td>${b.toFixed(4)}</td><td class="${cls}">${delta.toFixed(4)}</td><td class="delta-cell">${renderDeltaBar(delta, deltaScale)}</td><td>${spark}</td></tr>`;
  }
  html += '</table>';

  if (reportA.tag_breakdown && reportB.tag_breakdown) {
    const tagMapA = Object.fromEntries(reportA.tag_breakdown.map((t) => [t.tag, t]));
    const tagMapB = Object.fromEntries(reportB.tag_breakdown.map((t) => [t.tag, t]));
    const tags = Array.from(new Set([...Object.keys(tagMapA), ...Object.keys(tagMapB)])).sort();
    html += '<h3>Per-Tag Deltas</h3>';
    tags.forEach((tag) => {
      html += `<div class="card"><h4>${tag}</h4>`;
      html += '<table><tr><th>Metric</th><th>Run A</th><th>Run B</th><th>Delta</th><th>Spark</th></tr>';
      for (const metric of metrics) {
        const a = tagMapA[tag]?.metrics?.[metric] ?? 0;
        const b = tagMapB[tag]?.metrics?.[metric] ?? 0;
        const delta = b - a;
        const cls = delta >= 0 ? 'delta-positive' : 'delta-negative';
        html += `<tr><td>${metric}</td><td>${a.toFixed(4)}</td><td>${b.toFixed(4)}</td><td class="${cls}">${delta.toFixed(4)}</td><td class="delta-cell">${renderDeltaBar(delta, deltaScale)}</td></tr>`;
      }
      html += '</table></div>';
    });
  }

  compareOutput.innerHTML = html;
}

function computeDeltaScale(reportA, reportB, metrics) {
  let maxAbs = 0.001;
  for (const metric of metrics) {
    const a = reportA.summary?.[metric] ?? 0;
    const b = reportB.summary?.[metric] ?? 0;
    maxAbs = Math.max(maxAbs, Math.abs(b - a));
  }
  if (reportA.tag_breakdown && reportB.tag_breakdown) {
    const tagMapA = Object.fromEntries(reportA.tag_breakdown.map((t) => [t.tag, t]));
    const tagMapB = Object.fromEntries(reportB.tag_breakdown.map((t) => [t.tag, t]));
    const tags = new Set([...Object.keys(tagMapA), ...Object.keys(tagMapB)]);
    for (const tag of tags) {
      for (const metric of metrics) {
        const a = tagMapA[tag]?.metrics?.[metric] ?? 0;
        const b = tagMapB[tag]?.metrics?.[metric] ?? 0;
        maxAbs = Math.max(maxAbs, Math.abs(b - a));
      }
    }
  }
  return maxAbs;
}

function renderDeltaBar(delta, scale) {
  const width = Math.min(50, Math.round((Math.abs(delta) / scale) * 50));
  if (delta >= 0) {
    return `<div class="delta-bar"><span class="pos" style="width:${width}%;"></span></div>`;
  }
  return `<div class="delta-bar"><span class="neg" style="width:${width}%;"></span></div>`;
}

function buildTagSparkline(reportA, reportB, metric) {
  if (!reportA.tag_breakdown || !reportB.tag_breakdown) return '';
  const tagMapA = Object.fromEntries(reportA.tag_breakdown.map((t) => [t.tag, t]));
  const tagMapB = Object.fromEntries(reportB.tag_breakdown.map((t) => [t.tag, t]));
  const tags = Array.from(new Set([...Object.keys(tagMapA), ...Object.keys(tagMapB)])).sort();
  if (!tags.length) return '';
  const deltas = tags.map((tag) => (tagMapB[tag]?.metrics?.[metric] ?? 0) - (tagMapA[tag]?.metrics?.[metric] ?? 0));
  const maxAbs = Math.max(0.001, ...deltas.map((d) => Math.abs(d)));
  const bars = deltas
    .map((d) => {
      const height = Math.max(2, Math.round((Math.abs(d) / maxAbs) * 16));
      const color = d >= 0 ? '#1a7f37' : '#b42318';
      return `<span style="height:${height}px;background:${color};"></span>`;
    })
    .join('');
  return `<div class="sparkline" title="Tag deltas">${bars}</div>`;
}

function renderTagChips(tags) {
  tagChips.innerHTML = '';
  const allChip = createChip('All', true);
  tagChips.appendChild(allChip);
  tags.sort().forEach((tag) => tagChips.appendChild(createChip(tag, false)));
}

function createChip(label, active) {
  const chip = document.createElement('button');
  chip.className = `chip${active ? ' active' : ''}`;
  chip.textContent = label;
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip').forEach((el) => el.classList.remove('active'));
    chip.classList.add('active');
    tagFilter.value = label === 'All' ? '' : label;
    applyFilter();
  });
  return chip;
}

function applyFilter() {
  const query = tagFilter.value.toLowerCase().trim();
  document.querySelectorAll('#itemsTable tbody tr').forEach((row) => {
    const tags = row.dataset.tags || '';
    row.style.display = !query || tags.includes(query) ? '' : 'none';
  });
}

tagFilter.addEventListener('input', () => {
  document.querySelectorAll('.chip').forEach((el) => el.classList.remove('active'));
  applyFilter();
});

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function buildCsv(report) {
  const lines = [];
  lines.push('scope,metric,value');
  for (const metric of report.metrics || []) {
    const value = report.summary?.[metric] ?? 0;
    lines.push(`overall,${metric},${value.toFixed(4)}`);
  }
  if (report.tag_breakdown) {
    for (const tagInfo of report.tag_breakdown) {
      for (const metric of report.metrics || []) {
        const value = tagInfo.metrics?.[metric] ?? 0;
        lines.push(`tag:${tagInfo.tag},${metric},${value.toFixed(4)}`);
      }
    }
  }
  return lines.join('\n');
}

function buildHtml(report) {
  const rows = report.metrics
    .map((m) => `<tr><td>${escapeHtml(m)}</td><td>${(report.summary?.[m] ?? 0).toFixed(4)}</td></tr>`)
    .join('');
  let tagSection = '';
  if (report.tag_breakdown) {
    tagSection = report.tag_breakdown
      .map((tag) => {
        const tagRows = report.metrics
          .map((m) => `<tr><td>${escapeHtml(m)}</td><td>${(tag.metrics?.[m] ?? 0).toFixed(4)}</td></tr>`)
          .join('');
        return `<h3>Tag: ${escapeHtml(tag.tag)} (${tag.count})</h3><table><tr><th>Metric</th><th>Value</th></tr>${tagRows}</table>`;
      })
      .join('');
  }
  return `<!doctype html><html><head><meta charset="utf-8"><title>LLM Eval Report</title></head><body>
  <h1>LLM Eval Report</h1>
  <p>Run ID: ${escapeHtml(report.run_id || '')} | Model: ${escapeHtml(report.model_name || '')}</p>
  <h2>Summary</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>${rows}</table>
  ${tagSection}
  </body></html>`;
}

function downloadFile(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
