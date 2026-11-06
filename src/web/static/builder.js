const builderForm = document.getElementById("builderForm");
const projectListEl = document.getElementById("projectList");
const projectTitleEl = document.getElementById("projectTitle");
const projectMetaEl = document.getElementById("projectMeta");
const projectLogEl = document.getElementById("projectLog");
const trainProjectBtn = document.getElementById("trainProjectBtn");
const reviewDatasetBtn = document.getElementById("reviewDatasetBtn");
const refreshProjectsBtn = document.getElementById("refreshProjectsBtn");
const refreshStatusBtn = document.getElementById("refreshStatusBtn");
const refreshModelsBtn = document.getElementById("refreshModelsBtn");
const resetBuilderBtn = document.getElementById("resetBuilderBtn");
const openProjectChatLink = document.getElementById("openProjectChatLink");
const modelCatalogEl = document.getElementById("modelCatalog");
const datasetReviewContentEl = document.getElementById("datasetReviewContent");
const reviewBadgeEl = document.getElementById("reviewBadge");
const projectFamilySelect = document.getElementById("projectFamilySelect");
const profileSelect = document.getElementById("profileSelect");
const buildModeSelect = document.getElementById("buildModeSelect");
const predictiveTaskSelect = document.getElementById("predictiveTaskSelect");
const schemaFieldsWrap = document.getElementById("schemaFieldsWrap");
const labelFieldsWrap = document.getElementById("labelFieldsWrap");
const languageModelSection = document.getElementById("languageModelSection");
const predictiveSection = document.getElementById("predictiveSection");
const forecastHorizonWrap = document.getElementById("forecastHorizonWrap");
const nameInput = builderForm.querySelector('input[name="name"]');
const slugInput = builderForm.querySelector('input[name="slug"]');
const sequenceLengthInput = builderForm.querySelector('input[name="sequence_length"]');
const vocabSizeInput = builderForm.querySelector('input[name="vocab_size"]');
const modelSizeSelect = builderForm.querySelector('select[name="model_size"]');

const MODEL_SIZE_DEFAULTS = {
  safe_6m: { sequenceLength: 128, vocabSize: 3200 },
  balanced_11m: { sequenceLength: 192, vocabSize: 4000 },
  stretch_16m: { sequenceLength: 192, vocabSize: 4000 },
};

const state = {
  projects: window.BUILDER_BOOTSTRAP?.projects || [],
  models: window.BUILDER_BOOTSTRAP?.models || [],
  selectedSlug: null,
};

function slugify(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-+|-+$/g, "");
}

function statusLabel(status) {
  if (status === "running") return "Running";
  if (status === "completed") return "Completed";
  if (status === "failed") return "Failed";
  return "Idle";
}

function renderMeta(project) {
  const stats = project.stats || {};
  const metrics = project.evaluation_metrics || {};
  const items = [
    ["Status", statusLabel(project.status)],
    ["Family", project.project_family || "-"],
    ["Mode", project.build_mode || "-"],
    ["Profile", project.profile || "-"],
  ];

  if (project.project_family === "predictive_ai") {
    items.push(
      ["Task", project.predictive_task || "-"],
      ["Target", project.target_column || "-"],
      ["Time Column", project.time_column || "-"],
      ["Rows", String(stats.dataset_rows || project.dataset_overview?.rows || 0)],
      ["Columns", String(stats.dataset_columns || project.dataset_overview?.columns || 0)],
      ["Model Artifact", project.active_model || "-"]
    );
    Object.entries(metrics).forEach(([key, value]) => {
      items.push([key.toUpperCase(), String(value)]);
    });
  } else {
    items.push(
      ["Language", project.language_mode || "-"],
      ["Foundation", project.foundation_pack || "-"],
      ["Model Preset", project.model_size || "-"],
      ["Topics", (project.topics || []).join(", ") || "-"],
      ["Dialogs", String(stats.final_dialogues || 0)],
      ["Checkpoint", project.active_checkpoint || "-"]
    );
    if ((project.schema_fields || []).length) {
      items.push(["Schema", project.schema_fields.join(", ")]);
    }
    if ((project.label_options || []).length) {
      items.push(["Labels", project.label_options.join(", ")]);
    }
    if (project.scoring_scale) {
      items.push(["Score Scale", project.scoring_scale]);
    }
  }
  projectMetaEl.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="builder-meta-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}

function renderProjects() {
  if (!state.projects.length) {
    projectListEl.innerHTML = `<div class="builder-empty">No projects yet. Create the first one from the form.</div>`;
    return;
  }
  projectListEl.innerHTML = state.projects
    .map((project) => {
      const active = project.slug === state.selectedSlug ? "active" : "";
      return `
        <button class="builder-project-card ${active}" type="button" data-slug="${project.slug}">
          <div>
            <strong>${project.name}</strong>
            <p>${project.specialization || project.description || ""}</p>
            <small>${project.project_family === "predictive_ai" ? "Predictive AI" : "Language Model"}</small>
          </div>
          <span class="builder-status-pill status-${project.status || "idle"}">${statusLabel(project.status)}</span>
        </button>
      `;
    })
    .join("");

  projectListEl.querySelectorAll("[data-slug]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedSlug = button.dataset.slug;
      renderProjects();
      refreshSelectedProject();
    });
  });
}

function renderDatasetReview(review, project) {
  if (!project || project.project_family !== "predictive_ai") {
    reviewBadgeEl.textContent = "N/A";
    reviewBadgeEl.className = "builder-status-pill status-idle";
    datasetReviewContentEl.innerHTML = `<div class="builder-empty">Dataset review is available only for predictive AI projects.</div>`;
    reviewDatasetBtn.disabled = true;
    return;
  }

  reviewDatasetBtn.disabled = false;
  if (!review) {
    reviewBadgeEl.textContent = "Not Loaded";
    reviewBadgeEl.className = "builder-status-pill status-idle";
    datasetReviewContentEl.innerHTML = `<div class="builder-empty">Click "Review Dataset" to inspect missing values, target quality, cardinality, and readiness before training.</div>`;
    return;
  }

  const missingRows = (review.missing || [])
    .map((item) => `<tr><td>${item.column}</td><td>${item.missing}</td><td>${item.ratio}</td></tr>`)
    .join("");
  const warnings = (review.warnings || []).length
    ? `<ul class="builder-review-list">${review.warnings.map((item) => `<li>${item}</li>`).join("")}</ul>`
    : `<p class="builder-note">No major warnings were detected.</p>`;
  const recommendations = (review.recommendations || []).length
    ? `<ul class="builder-review-list">${review.recommendations.map((item) => `<li>${item}</li>`).join("")}</ul>`
    : `<p class="builder-note">No extra actions are required before training.</p>`;
  const targetSummary = Object.entries(review.target_summary || {})
    .map(([key, value]) => `<div class="builder-meta-card"><span>${key}</span><strong>${typeof value === "object" ? JSON.stringify(value) : value}</strong></div>`)
    .join("");
  const previewHeaders = review.preview?.length ? Object.keys(review.preview[0]) : [];
  const previewTable = review.preview?.length
    ? `
      <table class="predictive-table">
        <thead><tr>${previewHeaders.map((key) => `<th>${key}</th>`).join("")}</tr></thead>
        <tbody>
          ${review.preview.map((row) => `<tr>${previewHeaders.map((key) => `<td>${row[key] ?? "-"}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    `
    : `<div class="builder-empty">No preview rows.</div>`;

  reviewBadgeEl.textContent = `${review.readiness_label} (${review.readiness_score})`;
  reviewBadgeEl.className = `builder-status-pill ${review.readiness_score >= 85 ? "status-completed" : review.readiness_score >= 65 ? "status-running" : "status-failed"}`;
  datasetReviewContentEl.innerHTML = `
    <div class="builder-meta-grid">
      <div class="builder-meta-card"><span>Rows</span><strong>${review.rows}</strong></div>
      <div class="builder-meta-card"><span>Columns</span><strong>${review.columns}</strong></div>
      <div class="builder-meta-card"><span>Duplicates</span><strong>${review.duplicate_rows}</strong></div>
    </div>
    ${targetSummary ? `<div class="builder-meta-grid">${targetSummary}</div>` : ""}
    <div class="builder-review-grid">
      <article class="builder-review-card">
        <h3>Warnings</h3>
        ${warnings}
      </article>
      <article class="builder-review-card">
        <h3>Recommendations</h3>
        ${recommendations}
      </article>
    </div>
    <article class="builder-review-card">
      <h3>Missing Values</h3>
      ${missingRows ? `<table class="predictive-table"><thead><tr><th>Column</th><th>Missing</th><th>Ratio</th></tr></thead><tbody>${missingRows}</tbody></table>` : `<p class="builder-note">No missing values were detected.</p>`}
    </article>
    <article class="builder-review-card">
      <h3>Preview</h3>
      <div class="predictive-table-wrap">${previewTable}</div>
    </article>
  `;
}

async function loadDatasetReview() {
  if (!state.selectedSlug) {
    renderDatasetReview(null, null);
    return;
  }
  const project = state.projects.find((item) => item.slug === state.selectedSlug);
  if (!project || project.project_family !== "predictive_ai") {
    renderDatasetReview(null, project);
    return;
  }
  reviewBadgeEl.textContent = "Loading";
  reviewBadgeEl.className = "builder-status-pill status-running";
  const data = await fetchJson(`/api/builder/projects/${state.selectedSlug}/review`);
  renderDatasetReview(data.review, project);
}

function renderModelCatalog() {
  if (!state.models.length) {
    modelCatalogEl.innerHTML = `<div class="builder-empty">No models found yet.</div>`;
    return;
  }
  modelCatalogEl.innerHTML = state.models
    .map((model) => {
      const badge = model.is_default_builder_base ? `<span class="builder-status-pill status-completed">Builder Base</span>` : "";
      const originLabel = model.origin === "project" ? `Project: ${model.project || "-"}` : "Core checkpoint";
      return `
        <article class="builder-model-card">
          <div class="builder-model-head">
            <strong>${model.path}</strong>
            ${badge}
          </div>
          <p>${originLabel}</p>
          <p><span>Profile:</span> ${model.profile || "-"}</p>
          <p><span>Mode:</span> ${model.build_mode || "-"}</p>
          <p><span>Use:</span> ${model.recommended_for || "-"}</p>
        </article>
      `;
    })
    .join("");
}

function syncAdvancedFields() {
  const projectFamily = projectFamilySelect.value;
  const profile = profileSelect.value;
  const buildMode = buildModeSelect.value;
  const predictiveTask = predictiveTaskSelect.value;
  const showSchema = profile === "extraction_json";
  const showLabels = profile === "classification_scoring";
  const showPredictive = projectFamily === "predictive_ai";
  const showLanguage = !showPredictive;

  languageModelSection.classList.toggle("is-hidden", !showLanguage);
  predictiveSection.classList.toggle("is-hidden", !showPredictive);
  schemaFieldsWrap.classList.toggle("is-hidden", !showSchema);
  labelFieldsWrap.classList.toggle("is-hidden", !showLabels);
  forecastHorizonWrap.classList.toggle("is-hidden", !showPredictive || predictiveTask !== "forecasting");

  const preset = MODEL_SIZE_DEFAULTS[modelSizeSelect.value];
  if (showLanguage && buildMode === "from_scratch_tiny" && preset) {
    sequenceLengthInput.value = preset.sequenceLength;
    vocabSizeInput.value = preset.vocabSize;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "request failed");
  }
  return data;
}

async function loadProjects() {
  const data = await fetchJson("/api/builder/projects");
  state.projects = data.projects || [];
  if (!state.selectedSlug && state.projects.length) {
    state.selectedSlug = state.projects[0].slug;
  } else if (state.selectedSlug && !state.projects.find((project) => project.slug === state.selectedSlug)) {
    state.selectedSlug = state.projects[0]?.slug || null;
  }
  renderProjects();
  if (state.selectedSlug) {
    await refreshSelectedProject();
  } else {
    projectTitleEl.textContent = "בחר פרויקט";
    projectMetaEl.innerHTML = "";
    projectLogEl.textContent = "(no project selected)";
    renderDatasetReview(null, null);
  }
}

async function loadModels() {
  const data = await fetchJson("/api/builder/models");
  state.models = data.models || [];
  renderModelCatalog();
}

async function refreshSelectedProject() {
  if (!state.selectedSlug) {
    return;
  }
  const [projectData, logData] = await Promise.all([
    fetchJson(`/api/builder/projects/${state.selectedSlug}/status`),
    fetchJson(`/api/builder/projects/${state.selectedSlug}/logs`),
  ]);
  const project = projectData.project;
  state.projects = state.projects.map((item) => (item.slug === project.slug ? project : item));
  renderProjects();
  projectTitleEl.textContent = project.name;
  renderMeta(project);
  projectLogEl.textContent = [project.evaluation_summary || "", logData.log_tail || "(log is empty)"].filter(Boolean).join("\n\n");
  openProjectChatLink.href = project.project_family === "predictive_ai"
    ? `/builder/projects/${project.slug}/predictive`
    : `/builder/projects/${project.slug}/chat`;
  openProjectChatLink.textContent = project.project_family === "predictive_ai" ? "Open Dashboard" : "Open Chat";
  openProjectChatLink.classList.remove("disabled-link");
  await loadDatasetReview().catch(() => renderDatasetReview(null, project));
}

builderForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(builderForm);
  const data = await fetchJson("/api/builder/projects", {
    method: "POST",
    body: formData,
  });
  state.selectedSlug = data.project.slug;
  await loadProjects();
  await loadModels();
});

trainProjectBtn.addEventListener("click", async () => {
  if (!state.selectedSlug) {
    return;
  }
  await fetchJson(`/api/builder/projects/${state.selectedSlug}/train`, { method: "POST" });
  await refreshSelectedProject();
  await loadModels();
});

reviewDatasetBtn.addEventListener("click", async () => {
  await loadDatasetReview();
});

refreshProjectsBtn.addEventListener("click", async () => {
  await loadProjects();
});

refreshStatusBtn.addEventListener("click", async () => {
  await refreshSelectedProject();
});

refreshModelsBtn.addEventListener("click", async () => {
  await loadModels();
});

resetBuilderBtn.addEventListener("click", () => {
  builderForm.reset();
  syncAdvancedFields();
});

nameInput.addEventListener("input", () => {
  if (!slugInput.value.trim()) {
    slugInput.value = slugify(nameInput.value);
  }
});

profileSelect.addEventListener("change", syncAdvancedFields);
buildModeSelect.addEventListener("change", syncAdvancedFields);
modelSizeSelect.addEventListener("change", syncAdvancedFields);
projectFamilySelect.addEventListener("change", syncAdvancedFields);
predictiveTaskSelect.addEventListener("change", syncAdvancedFields);

setInterval(() => {
  if (state.selectedSlug) {
    refreshSelectedProject().catch(() => {});
  }
}, 5000);

syncAdvancedFields();
loadProjects()
  .then(loadModels)
  .catch((error) => {
    projectLogEl.textContent = `Error: ${error.message}`;
  });
