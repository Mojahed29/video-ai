// Tiny DOM helpers — keeps the rest of the code declarative and framework-free.
export const $ = (id) => document.getElementById(id);

export const show = (el) => el && el.classList.remove("hidden");
export const hide = (el) => el && el.classList.add("hidden");
export const toggle = (el, on) => el && el.classList.toggle("hidden", !on);

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

export function fileExtension(name = "") {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

// Escape user/server-controlled strings before inserting into innerHTML.
export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
