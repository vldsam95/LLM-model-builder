const predictiveBootstrap = window.PREDICTIVE_BOOTSTRAP || {};
const predictiveProject = predictiveBootstrap.project || {};
const predictiveEvaluation = predictiveBootstrap.evaluation || {};

const predictiveMetricsEl = document.getElementById("predictiveMetrics");
const featureImportanceEl = document.getElementById("featureImportance");
const predictiveChartEl = document.getElementById("predictiveChart");
const confusionWrapEl = document.getElementById("confusionWrap");
const confusionMatrixEl = document.getElementById("confusionMatrix");
const predictiveFormEl = document.getElementById("predictiveForm");
const predictiveInputGridEl = document.getElementById("predictiveInputGrid");
const predictiveResultEl = document.getElementById("predictiveResult");
const predictionSamplesEl = document.getElementById("predictionSamples");
const forecastWrapEl = document.getElementById("forecastWrap");
const forecastTableEl = document.getElementById("forecastTable");
const predictiveRunBtn = document.getElementById("predictiveRunBtn");

function formatMetricName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .toUpperCase();
}

function renderMetrics() {
  const metrics = predictiveEvaluation.metrics || {};
  const entries = Object.entries(metrics);
  if (!entries.length) {
    predictiveMetricsEl.innerHTML = `<div class="builder-empty">No metrics yet. Train the project first.</div>`;
    return;
  }
  predictiveMetricsEl.innerHTML = entries
    .map(
      ([key, value]) => `
        <article class="predictive-metric-card">
          <span>${formatMetricName(key)}</span>
          <strong>${value}</strong>
        </article>
      `
    )
    .join("");
}

function renderImportance() {
  const rows = predictiveEvaluation.feature_importance || [];
  if (!rows.length) {
    featureImportanceEl.innerHTML = `<div class="builder-empty">Feature importance will appear after training.</div>`;
    return;
  }
  const maxValue = Math.max(...rows.map((item) => Number(item.importance) || 0), 1e-6);
  featureImportanceEl.innerHTML = rows
    .map((item) => {
      const width = Math.max(6, Math.round(((Number(item.importance) || 0) / maxValue) * 100));
      return `
        <div class="predictive-importance-row">
          <div class="predictive-importance-head">
            <strong>${item.feature}</strong>
            <span>${item.importance}</span>
          </div>
          <div class="predictive-importance-bar">
            <span style="width:${width}%"></span>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderSvgChart(points) {
  if (!points.length) {
    predictiveChartEl.innerHTML = `<div class="builder-empty">No chart data yet.</div>`;
    return;
  }

  const width = 720;
  const height = 280;
  const padding = 28;
  const numericPoints = points.filter((point) => Number.isFinite(Number(point.actual)) && Number.isFinite(Number(point.predicted)));
  if (!numericPoints.length) {
    predictiveChartEl.innerHTML = `<div class="builder-empty">Chart available for numeric targets only.</div>`;
    return;
  }
  const values = numericPoints.flatMap((point) => [Number(point.actual), Number(point.predicted)]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xStep = numericPoints.length > 1 ? (width - padding * 2) / (numericPoints.length - 1) : 1;
  const yScale = (value) => {
    if (max === min) {
      return height / 2;
    }
    return height - padding - ((value - min) / (max - min)) * (height - padding * 2);
  };
  const buildPath = (key) =>
    numericPoints
      .map((point, index) => `${index === 0 ? "M" : "L"} ${padding + index * xStep} ${yScale(Number(point[key]))}`)
      .join(" ");

  predictiveChartEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="predictive-chart" role="img" aria-label="Actual versus predicted chart">
      <path d="${buildPath("actual")}" class="predictive-chart-line actual"></path>
      <path d="${buildPath("predicted")}" class="predictive-chart-line predicted"></path>
    </svg>
    <div class="predictive-chart-legend">
      <span><i class="actual"></i>Actual</span>
      <span><i class="predicted"></i>Predicted</span>
    </div>
  `;
}

function renderConfusion() {
  const labels = predictiveEvaluation.labels || [];
  const matrix = predictiveEvaluation.confusion_matrix || [];
  if (!labels.length || !matrix.length) {
    confusionWrapEl.classList.add("is-hidden");
    return;
  }
  confusionWrapEl.classList.remove("is-hidden");
  const header = labels.map((label) => `<th>${label}</th>`).join("");
  const body = matrix
    .map(
      (row, rowIndex) => `
        <tr>
          <th>${labels[rowIndex]}</th>
          ${row.map((cell) => `<td>${cell}</td>`).join("")}
        </tr>
      `
    )
    .join("");
  confusionMatrixEl.innerHTML = `
    <table class="predictive-table">
      <thead>
        <tr>
          <th>Actual \\ Predicted</th>
          ${header}
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderSamples() {
  const rows = predictiveEvaluation.sample_predictions || [];
  if (!rows.length) {
    predictionSamplesEl.innerHTML = `<div class="builder-empty">No validation examples yet.</div>`;
    return;
  }

  const headers = ["actual", "predicted", "time"];
  predictionSamplesEl.innerHTML = `
    <table class="predictive-table">
      <thead>
        <tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .slice(0, 12)
          .map(
            (row) => `
              <tr>
                <td>${row.actual ?? "-"}</td>
                <td>${row.predicted ?? "-"}</td>
                <td>${row.time ?? "-"}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderForecast() {
  const rows = predictiveEvaluation.future_forecast || [];
  if (!rows.length) {
    forecastWrapEl.classList.add("is-hidden");
    return;
  }
  forecastWrapEl.classList.remove("is-hidden");
  forecastTableEl.innerHTML = `
    <table class="predictive-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Predicted</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td>${row.time}</td>
                <td>${row.predicted}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function buildInputControl(field) {
  const inputType = field.kind === "number" ? "number" : "text";
  const step = field.kind === "number" ? 'step="any"' : "";
  const placeholder = field.example != null ? String(field.example) : "";
  return `
    <label>
      <span>${field.name}</span>
      <input data-feature-name="${field.name}" type="${inputType}" ${step} placeholder="${placeholder}">
    </label>
  `;
}

function renderPredictionForm() {
  const task = predictiveEvaluation.task || predictiveProject.predictive_task;
  const schema = predictiveEvaluation.input_schema || [];
  if (task === "forecasting") {
    predictiveInputGridEl.innerHTML = `<div class="builder-empty">Forecasting projects expose a ready-made future forecast below. Manual row prediction is not required.</div>`;
    predictiveRunBtn.textContent = "Refresh Forecast";
    return;
  }
  if (!schema.length) {
    predictiveInputGridEl.innerHTML = `<div class="builder-empty">Input schema will appear after training.</div>`;
    predictiveRunBtn.disabled = true;
    return;
  }
  predictiveInputGridEl.innerHTML = schema.map(buildInputControl).join("");
}

function collectPredictiveFeatures() {
  const payload = {};
  predictiveInputGridEl.querySelectorAll("[data-feature-name]").forEach((input) => {
    if (!input.value.trim()) {
      return;
    }
    payload[input.dataset.featureName] = input.type === "number" ? Number(input.value) : input.value.trim();
  });
  return payload;
}

async function runPrediction(event) {
  event.preventDefault();
  predictiveRunBtn.disabled = true;
  try {
    const response = await fetch(`/api/builder/projects/${predictiveProject.slug}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: collectPredictiveFeatures() }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "prediction failed");
    }
    predictiveResultEl.textContent = JSON.stringify(data.response, null, 2);
  } catch (error) {
    predictiveResultEl.textContent = `Error: ${error.message}`;
  } finally {
    predictiveRunBtn.disabled = false;
  }
}

renderMetrics();
renderImportance();
renderSvgChart(predictiveEvaluation.chart_data || []);
renderConfusion();
renderSamples();
renderForecast();
renderPredictionForm();

if (predictiveFormEl) {
  predictiveFormEl.addEventListener("submit", runPrediction);
}

