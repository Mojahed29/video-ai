// SVG / markup builders for the score gauge and the explainability timelines.
import { BAND_HIGH, BAND_LOW } from "./config.js";
import { escapeHtml } from "./dom.js";

// Banded color for verdicts/values (matches the legend + tokens).
export function scoreColor(score) {
  if (score == null) return "var(--muted)";
  if (score < BAND_LOW) return "var(--score-low)";
  if (score < BAND_HIGH) return "var(--score-mid)";
  return "var(--score-high)";
}

// Continuous color for the heat strip. Interpolating in HSL (green hue -> red
// hue) keeps the ramp vivid and avoids the muddy olive that sRGB interpolation
// produces between green and orange.
export function heatColor(score) {
  const s = Math.max(0, Math.min(100, score ?? 50));
  const hue = 142 - (142 - 6) * (s / 100); // 142° green -> 6° red
  return hslToRgb(hue, 0.62, 0.4);
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] =
    h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : [0, x, c];
  const to = (v) => Math.round((v + m) * 255);
  return `rgb(${to(r)}, ${to(g)}, ${to(b)})`;
}

// Semicircular arc gauge (0–100). Self-contained SVG with the number inside.
export function gaugeSVG(value) {
  const v = Math.max(0, Math.min(100, Math.round(value)));
  const r = 84;
  const cx = 100;
  const cy = 104;
  const len = Math.PI * r; // length of the semicircle
  const dash = (v / 100) * len;
  const color = scoreColor(v);
  const track = "var(--surface-sunken)";
  const path = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  return `
    <svg class="gauge" viewBox="0 0 200 132" role="img"
         aria-label="Overall likelihood ${v} out of 100">
      <path d="${path}" fill="none" stroke="${track}" stroke-width="16" stroke-linecap="round"/>
      <path d="${path}" fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round"
            stroke-dasharray="${dash.toFixed(1)} ${len.toFixed(1)}"/>
      <text x="100" y="96" text-anchor="middle" class="gauge-number" fill="${color}">${v}</text>
      <text x="100" y="120" text-anchor="middle" class="gauge-unit" fill="var(--muted)">/ 100</text>
    </svg>`;
}

// Per-frame heat strip: one cell per sampled frame, colored by P(fake).
export function heatStrip(timeline) {
  if (!timeline || !timeline.length) return "";
  const cells = timeline
    .map((p) => {
      const label = `${p.t_start}s · ${Math.round(p.score)}/100${p.note ? ` · ${p.note}` : ""}`;
      return `<span class="heatcell" style="background:${heatColor(p.score)}" title="${escapeHtml(label)}"></span>`;
    })
    .join("");
  return `<div class="heatstrip" role="img" aria-label="${escapeHtml(strfrom(timeline, "frame"))}">${cells}</div>`;
}

// Audio timeline: a labeled bar per scored window.
export function timelineBars(timeline) {
  if (!timeline || !timeline.length) return "";
  const rows = timeline
    .map((p) => {
      const w = Math.max(2, Math.min(100, p.score));
      return `
        <div class="tl-row">
          <span class="tl-time">${fmt(p.t_start)}–${fmt(p.t_end)}s</span>
          <span class="tl-track"><span class="tl-fill" style="width:${w}%;background:${heatColor(p.score)}"></span></span>
          <span class="tl-val" style="color:${scoreColor(p.score)}">${Math.round(p.score)}</span>
        </div>`;
    })
    .join("");
  return `<div class="timeline-bars" role="img" aria-label="${escapeHtml(strfrom(timeline, "window"))}">${rows}</div>`;
}

export function scaleLegend() {
  return `
    <div class="scale-legend" aria-hidden="true">
      <span>Authentic</span>
      <span class="legend-grad"></span>
      <span>AI-generated</span>
    </div>`;
}

// Accessible text summary of a timeline (peak moment).
function strfromPeak(timeline) {
  return timeline.reduce((a, b) => (b.score > a.score ? b : a), timeline[0]);
}
function strfrom(timeline, unit) {
  const peak = strfromPeak(timeline);
  return `${timeline.length} ${unit}s scored. Peak likelihood ${Math.round(peak.score)} of 100 at ${peak.t_start} seconds.`;
}

function fmt(n) {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}
