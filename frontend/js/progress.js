// Staged progress controller.
//
// Upload progress is REAL (XHR upload events). The analysis itself is a single
// monolithic request, so its sub-stages (extract / visual / audio / fuse) are
// time-estimated: we advance the highlighted stage on a timer and hold on the
// last one until the server actually responds. This honestly communicates what
// the server is doing without faking a precise percentage.
import { ANALYSIS_STAGES, UPLOAD_STAGE } from "./config.js";
import { $ } from "./dom.js";

const ALL_STAGES = [UPLOAD_STAGE, ...ANALYSIS_STAGES];

const CHECK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`;

export class ProgressController {
  constructor() {
    this.titleEl = $("progress-title");
    this.pctEl = $("progress-pct");
    this.trackEl = $("progress-track");
    this.fillEl = $("progress-fill");
    this.listEl = $("stage-list");
    this._timers = [];
    this._renderStages();
  }

  _renderStages() {
    this.listEl.innerHTML = ALL_STAGES.map(
      (s) => `
        <li class="stage" data-stage="${s.id}">
          <span class="stage-dot" aria-hidden="true"></span>
          <span class="stage-text">${s.label}</span>
        </li>`
    ).join("");
  }

  reset() {
    this._clearTimers();
    this.trackEl.classList.remove("is-indeterminate");
    this.fillEl.style.width = "0%";
    this.pctEl.textContent = "";
    this.trackEl.setAttribute("aria-valuenow", "0");
    this.listEl.querySelectorAll(".stage").forEach((el) => el.className = "stage");
    this._setActive("upload");
    this.titleEl.textContent = "Uploading video…";
  }

  // pct in [0, 1] — real upload progress.
  setUpload(pct) {
    const value = Math.round(pct * 100);
    this.fillEl.style.width = `${value}%`;
    this.pctEl.textContent = `${value}%`;
    this.trackEl.setAttribute("aria-valuenow", String(value));
    this.titleEl.textContent = "Uploading video…";
  }

  // Upload finished; begin the estimated analysis staging.
  beginAnalysis() {
    this._clearTimers();
    this._markDone("upload");
    this.pctEl.textContent = "";
    this.trackEl.removeAttribute("aria-valuenow");
    this.trackEl.classList.add("is-indeterminate");
    this.titleEl.textContent = "Analyzing… this can take up to a minute on CPU";

    let delay = 300;
    ANALYSIS_STAGES.forEach((stage, i) => {
      this._timers.push(
        setTimeout(() => {
          // Mark previous analysis stage done, activate this one.
          if (i > 0) this._markDone(ANALYSIS_STAGES[i - 1].id);
          this._setActive(stage.id);
        }, delay)
      );
      // The last stage is held (its full duration is not scheduled to "done").
      delay += stage.approxMs;
    });
  }

  finish() {
    this._clearTimers();
    ALL_STAGES.forEach((s) => this._markDone(s.id));
    this.trackEl.classList.remove("is-indeterminate");
    this.fillEl.style.width = "100%";
    this.titleEl.textContent = "Done";
  }

  _setActive(id) {
    const el = this.listEl.querySelector(`[data-stage="${id}"]`);
    if (!el || el.classList.contains("is-done")) return;
    el.classList.add("is-active");
    const dot = el.querySelector(".stage-dot");
    if (dot) dot.innerHTML = `<span class="spinner"></span>`;
  }

  _markDone(id) {
    const el = this.listEl.querySelector(`[data-stage="${id}"]`);
    if (!el) return;
    el.classList.remove("is-active");
    el.classList.add("is-done");
    const dot = el.querySelector(".stage-dot");
    if (dot) dot.innerHTML = CHECK;
  }

  _clearTimers() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }
}
