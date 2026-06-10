// Dropzone: drag & drop, click, and keyboard file selection + client-side
// validation (so obviously-invalid files are rejected before a 1-minute upload).
import { ACCEPTED_EXTENSIONS, MAX_FILE_BYTES } from "./config.js";
import { $, fileExtension, formatBytes } from "./dom.js";

export function initDropzone({ onSelect, onError }) {
  const zone = $("dropzone");
  const input = $("file-input");

  const openPicker = () => {
    if (zone.getAttribute("aria-disabled") === "true") return;
    input.click();
  };

  zone.addEventListener("click", openPicker);
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
      e.preventDefault();
      openPicker();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      if (zone.getAttribute("aria-disabled") !== "true") zone.classList.add("is-dragover");
    })
  );
  ["dragleave", "dragend", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("is-dragover");
    })
  );

  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) handle(file);
  });
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) handle(file);
    input.value = ""; // allow re-selecting the same file later
  });

  function handle(file) {
    const problem = validate(file);
    if (problem) onError(problem);
    else onSelect(file);
  }

  return {
    setDisabled(disabled) {
      zone.setAttribute("aria-disabled", String(disabled));
      zone.tabIndex = disabled ? -1 : 0;
    },
  };
}

function validate(file) {
  const ext = fileExtension(file.name);
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return {
      title: "Unsupported file type",
      message: `“${file.name}” isn’t a supported video. Please use ${ACCEPTED_EXTENSIONS.join(", ")}.`,
    };
  }
  if (file.size > MAX_FILE_BYTES) {
    return {
      title: "File is too large",
      message: `That file is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_FILE_BYTES)}.`,
    };
  }
  if (file.size === 0) {
    return { title: "Empty file", message: "That file appears to be empty." };
  }
  return null;
}
