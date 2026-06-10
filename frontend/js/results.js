// Renders the analysis result: overall gauge + verdict, the two sub-scores,
// the explainability timelines, and run metadata. All server-supplied strings
// are escaped before being placed in innerHTML.
import { BAND_HIGH, BAND_LOW } from "./config.js";
import { escapeHtml } from "./dom.js";
import { gaugeSVG, heatStrip, scaleLegend, scoreColor, timelineBars } from "./charts.js";

const ICON = {
  visual: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>`,
  audio: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h2l2 5 4-14 3 9 2-3h5"/></svg>`,
};

export function renderResult(el, data, { onReset } = {}) {
  el.innerHTML = [
    scorePanel(data),
    subscoresCard(data),
    explainCard(data),
    metaCard(data),
    `<div class="file-actions"><button type="button" class="btn btn-primary" id="reset-btn">Analyze another video</button></div>`,
  ].join("");

  const reset = el.querySelector("#reset-btn");
  if (reset && onReset) reset.addEventListener("click", onReset);
}

function verdict(score) {
  if (score < BAND_LOW)
    return ["Likely authentic", "Few signals of AI generation were found in this clip."];
  if (score < BAND_HIGH)
    return ["Mixed signals", "The detectors were uncertain — evidence points in both directions."];
  return ["Likely AI-generated", "Patterns consistent with AI generation or face-swapping were flagged."];
}

const CONFIDENCE_NOTE = {
  high: "Both tracks were assessed.",
  medium: "Only one track could be assessed.",
  low: "Neither track could be assessed.",
};

function scorePanel(data) {
  const [title, sub] = verdict(data.overall_score);
  const color = scoreColor(data.overall_score);
  const calibrated = data.calibrated
    ? `<span class="badge neutral" title="Scores were shrunk toward 50 to reflect limited signal.">calibrated</span>`
    : "";
  return `
    <div class="card score-panel reveal">
      <div class="gauge-wrap">${gaugeSVG(data.overall_score)}</div>
      <div class="score-meaning">
        <div class="verdict" style="color:${color}">${title}</div>
        <p class="verdict-sub">${escapeHtml(sub)}</p>
        <div class="band-row">
          <span class="badge ${escapeHtml(data.confidence_band)}">${escapeHtml(data.confidence_band)} confidence</span>
          ${calibrated}
          <span class="score-caption">${escapeHtml(CONFIDENCE_NOTE[data.confidence_band] || "")}</span>
        </div>
        <p class="score-caption">${escapeHtml(data.fusion_method)}</p>
      </div>
    </div>`;
}

function subscoresCard(data) {
  return `
    <div class="card reveal">
      <div class="subscores">
        ${subscore("Visual track", ICON.visual, data.visual)}
        ${subscore("Audio track", ICON.audio, data.audio)}
      </div>
    </div>`;
}

function subscore(label, icon, m) {
  const assessed = m.status === "assessed" && m.score != null;
  const color = assessed ? scoreColor(m.score) : "var(--muted)";
  const value = assessed
    ? `${m.score.toFixed(1)}<span class="gauge-unit"> / 100</span>`
    : `<span class="na">Not assessed</span>`;
  const meter = assessed
    ? `<div class="meter"><div class="meter-fill" style="width:${Math.max(2, Math.min(100, m.score))}%;background:${color}"></div></div>`
    : "";
  const detail = assessed ? m.detail : m.reason;
  const rawNote =
    assessed && m.raw_score != null
      ? `<p class="subscore-detail">Raw model probability <strong>${m.raw_score}%</strong>${
          m.raw_score !== m.score ? ` → calibrated <strong>${m.score}%</strong>` : ""
        }.</p>`
      : "";
  const model = m.model_used ? `<p class="subscore-model">Model: ${escapeHtml(m.model_used)}</p>` : "";
  return `
    <div class="subscore ${assessed ? "" : "not-assessed"}">
      <div class="subscore-head">
        <span class="subscore-label">${icon}${label}</span>
        ${assessed ? "" : `<span class="badge neutral">not assessed</span>`}
      </div>
      <div class="subscore-value" style="color:${color}">${value}</div>
      ${meter}
      ${detail ? `<p class="subscore-detail">${escapeHtml(detail)}</p>` : ""}
      ${rawNote}
      ${model}
    </div>`;
}

function explainCard(data) {
  const hasVisual = data.visual.timeline && data.visual.timeline.length;
  const hasAudio = data.audio.timeline && data.audio.timeline.length;
  if (!hasVisual && !hasAudio) return "";

  const visualPanel = `
    <div>
      <div class="panel-head"><span class="panel-title">Per-frame likelihood</span><span class="panel-hint">visual · over time</span></div>
      ${hasVisual ? heatStrip(data.visual.timeline) + scaleLegend() : notAssessed(data.visual.reason)}
    </div>`;
  const audioPanel = `
    <div>
      <div class="panel-head"><span class="panel-title">Audio timeline</span><span class="panel-hint">windowed</span></div>
      ${hasAudio ? timelineBars(data.audio.timeline) : notAssessed(data.audio.reason)}
    </div>`;
  return `
    <div class="card reveal">
      <div class="explain">${visualPanel}${audioPanel}</div>
    </div>`;
}

function notAssessed(reason) {
  return `<p class="subscore-detail">Not assessed${reason ? ` — ${escapeHtml(reason)}` : "."}</p>`;
}

function metaCard(data) {
  const warnings = (data.warnings || [])
    .map(
      (w) =>
        `<li><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg><span>${escapeHtml(w)}</span></li>`
    )
    .join("");
  return `
    <div class="card meta reveal">
      <div class="meta-grid">
        <span class="meta-item"><strong>Frames analyzed:</strong> ${escapeHtml(data.frames_analyzed)}</span>
        <span class="meta-item"><strong>Duration:</strong> ${escapeHtml(data.duration_seconds)}s</span>
        <span class="meta-item"><strong>Fusion:</strong> ${escapeHtml(data.fusion_method)}</span>
      </div>
      ${warnings ? `<ul class="warnings">${warnings}</ul>` : ""}
    </div>`;
}
