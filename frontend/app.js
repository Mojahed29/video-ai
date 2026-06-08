const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const progressSection = document.getElementById("progress");
const progressText = document.getElementById("progress-text");
const errorSection = document.getElementById("error");
const resultSection = document.getElementById("result");

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});

function resetViews() {
  errorSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  progressSection.classList.remove("hidden");
}

function showError(message) {
  progressSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorSection.textContent = message;
  errorSection.classList.remove("hidden");
}

async function handleFile(file) {
  resetViews();
  progressText.textContent = `Uploading "${file.name}" and analyzing — this can take a minute on CPU…`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || `Request failed with status ${response.status}`;
      showError(detail);
      return;
    }
    renderResult(data);
  } catch (err) {
    showError(`Network error while contacting the backend: ${err.message}`);
  } finally {
    fileInput.value = "";
  }
}

function scoreColor(score) {
  if (score === null || score === undefined) return "var(--muted)";
  if (score < 35) return "var(--good)";
  if (score < 65) return "var(--warn)";
  return "var(--bad)";
}

function modalityLine(modality) {
  if (modality.status !== "assessed") {
    return { value: "N/A", detail: modality.reason || "Not assessed.", model: "" };
  }
  return {
    value: `${modality.score.toFixed(1)}`,
    detail: modality.detail || "",
    model: modality.model_used ? `Model: ${modality.model_used}` : "",
  };
}

function renderResult(data) {
  progressSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  resultSection.classList.remove("hidden");

  const scoreEl = document.getElementById("score-number");
  scoreEl.textContent = Math.round(data.overall_score);
  scoreEl.style.color = scoreColor(data.overall_score);

  const badge = document.getElementById("confidence-band");
  badge.textContent = `${data.confidence_band} confidence`;
  badge.className = `badge ${data.confidence_band}`;

  const visual = modalityLine(data.visual);
  document.getElementById("visual-score").textContent = visual.value;
  document.getElementById("visual-score").style.color = scoreColor(data.visual_subscore);
  document.getElementById("visual-detail").textContent = visual.detail;
  document.getElementById("visual-model").textContent = visual.model;

  const audio = modalityLine(data.audio);
  document.getElementById("audio-score").textContent = audio.value;
  document.getElementById("audio-score").style.color = scoreColor(data.audio_subscore);
  document.getElementById("audio-detail").textContent = audio.detail;
  document.getElementById("audio-model").textContent = audio.model;

  document.getElementById("fusion-method").textContent = `Fusion: ${data.fusion_method}`;
  document.getElementById("frames-info").textContent =
    `Analyzed ${data.frames_analyzed} sampled frame(s) from a ${data.duration_seconds}s clip.`;

  const warningsEl = document.getElementById("warnings");
  warningsEl.innerHTML = "";
  (data.warnings || []).forEach((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    warningsEl.appendChild(li);
  });

  renderExplainability(data);
}

function renderExplainability(data) {
  // --- Visual: temporal-consistency note + occlusion saliency heatmap ---
  const visualExplain = document.getElementById("visual-explain");
  const temporal = data.visual?.temporal_consistency;
  const saliency = data.visual?.saliency;
  const temporalEl = document.getElementById("visual-temporal");
  const saliencyImg = document.getElementById("visual-saliency-img");
  const saliencyCaption = document.getElementById("visual-saliency-caption");

  temporalEl.textContent = temporal
    ? `Frame-to-frame consistency: inconsistency score ${temporal.inconsistency_score} ` +
      `(coefficient of variation ${temporal.coefficient_of_variation}). ${temporal.note}`
    : "";

  if (saliency?.image_base64) {
    saliencyImg.src = saliency.image_base64;
    saliencyImg.classList.remove("hidden");
    saliencyCaption.textContent = saliency.note || "";
  } else {
    saliencyImg.removeAttribute("src");
    saliencyImg.classList.add("hidden");
    saliencyCaption.textContent = "";
  }

  visualExplain.classList.toggle("hidden", !temporal && !saliency);

  // --- Audio: per-segment timeline ---
  const audioExplain = document.getElementById("audio-explain");
  const timelineEl = document.getElementById("audio-timeline");
  const timeline = data.audio?.timeline;
  timelineEl.innerHTML = "";

  if (timeline && timeline.length) {
    timeline.forEach((segment) => {
      const bar = document.createElement("div");
      bar.className = "timeline-bar";
      const pct = segment.score ?? 0;
      bar.style.height = `${Math.max(6, pct)}%`;
      bar.style.background = scoreColor(segment.score);
      bar.title = `${segment.start_seconds}s–${segment.end_seconds}s: ` +
        (segment.score === null ? "N/A" : `${segment.score.toFixed(1)}`);
      timelineEl.appendChild(bar);
    });
    audioExplain.classList.remove("hidden");
  } else {
    audioExplain.classList.add("hidden");
  }
}
