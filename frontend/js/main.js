// Entry point: a small explicit state machine
// (idle -> selected -> working -> success | error) that owns the views and
// wires the dropzone, progress controller, API call, and results renderer.
import { startAnalysis } from "./api.js";
import { $, formatBytes, hide, show } from "./dom.js";
import { initDropzone } from "./dropzone.js";
import { ProgressController } from "./progress.js";
import { renderResult } from "./results.js";

const els = {
  dropzone: $("dropzone"),
  fileCard: $("file-card"),
  progress: $("progress"),
  error: $("error"),
  result: $("result"),
  fileName: $("file-name"),
  fileSub: $("file-sub"),
  errorTitle: $("error-title"),
  errorMessage: $("error-message"),
};

const progress = new ProgressController();
let dropzoneApi;
let currentFile = null;
let currentAnalysis = null;

// Make the dynamic regions focus targets for clean keyboard/SR transitions.
els.error.tabIndex = -1;
els.result.tabIndex = -1;

function setState(state) {
  hide(els.dropzone);
  hide(els.fileCard);
  hide(els.progress);
  hide(els.error);
  hide(els.result);

  if (state === "idle") show(els.dropzone);
  if (state === "selected") show(els.fileCard);
  if (state === "working") show(els.progress);
  if (state === "error") show(els.error);
  if (state === "success") show(els.result);

  dropzoneApi?.setDisabled(state === "working");
}

function goIdle() {
  currentFile = null;
  currentAnalysis = null;
  setState("idle");
  els.dropzone.focus?.();
}

function selectFile(file) {
  currentFile = file;
  els.fileName.textContent = file.name;
  els.fileSub.textContent = `${formatBytes(file.size)} · ${file.type || "video"}`;
  setState("selected");
  $("analyze-btn").focus();
}

function showError({ title, message, info = false }) {
  els.errorTitle.textContent = title || "Something went wrong";
  els.errorMessage.textContent = message || "";
  els.error.classList.toggle("is-info", Boolean(info));
  setState("error");
  els.error.focus();
}

function startWorking() {
  if (!currentFile) return;
  setState("working");
  progress.reset();

  let analysisBegun = false;
  currentAnalysis = startAnalysis(currentFile, {
    onUploadProgress: (pct) => {
      progress.setUpload(pct);
      if (pct >= 0.999 && !analysisBegun) {
        analysisBegun = true;
        progress.beginAnalysis();
      }
    },
  });

  // Fallback: if upload-progress events never fired, still switch to analysis UI.
  const fallbackTimer = setTimeout(() => {
    if (!analysisBegun) {
      analysisBegun = true;
      progress.beginAnalysis();
    }
  }, 1200);

  currentAnalysis.promise
    .then((data) => {
      clearTimeout(fallbackTimer);
      progress.finish();
      setTimeout(() => {
        renderResult(els.result, data, { onReset: goIdle });
        setState("success");
        els.result.focus();
      }, 300);
    })
    .catch((err) => {
      clearTimeout(fallbackTimer);
      if (err && err.kind === "aborted") return; // user cancelled -> handled in cancel
      showError(mapError(err));
    });
}

function cancel() {
  currentAnalysis?.abort();
  currentAnalysis = null;
  if (currentFile) selectFile(currentFile);
  else goIdle();
}

function mapError(err) {
  const status = err?.status;
  const message = err?.message || "Unexpected error.";
  if (err?.kind === "network") return { title: "Can’t reach the server", message };
  if (err?.kind === "parse") return { title: "Unexpected response", message };
  const titles = {
    400: "Unsupported file",
    413: "File too large or too long",
    422: "Couldn’t read the video",
    500: "Analysis failed",
  };
  return { title: titles[status] || "Request failed", message };
}

function init() {
  dropzoneApi = initDropzone({ onSelect: selectFile, onError: showError });
  $("analyze-btn").addEventListener("click", startWorking);
  $("change-btn").addEventListener("click", goIdle);
  $("cancel-btn").addEventListener("click", cancel);
  $("retry-btn").addEventListener("click", goIdle);
  goIdle();
}

init();
