// Upload + analyze. Uses XMLHttpRequest (not fetch) for two reasons the UX
// needs: real upload-progress events, and a reliable abort() for "Cancel".
import { ANALYZE_ENDPOINT } from "./config.js";

/**
 * Start an analysis request.
 * @returns {{ promise: Promise<object>, abort: () => void }}
 *   The promise resolves with the parsed AnalyzeResponse, or rejects with an
 *   Error carrying `.status` (HTTP status) and `.kind` ("http" | "network" |
 *   "aborted" | "parse").
 */
export function startAnalysis(file, { onUploadProgress } = {}) {
  const xhr = new XMLHttpRequest();
  const form = new FormData();
  form.append("file", file);

  const promise = new Promise((resolve, reject) => {
    xhr.open("POST", ANALYZE_ENDPOINT);
    xhr.responseType = "text";

    if (xhr.upload && typeof onUploadProgress === "function") {
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) onUploadProgress(e.loaded / e.total);
      });
    }

    xhr.addEventListener("load", () => {
      let data = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        data = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (data) resolve(data);
        else reject(makeError("Received an unreadable response from the server.", { kind: "parse", status: xhr.status }));
        return;
      }
      const detail = data && data.detail ? data.detail : `Request failed (HTTP ${xhr.status}).`;
      reject(makeError(detail, { kind: "http", status: xhr.status }));
    });

    xhr.addEventListener("error", () =>
      reject(makeError("Could not reach the analysis server. Check that the backend is running.", { kind: "network" }))
    );
    xhr.addEventListener("abort", () =>
      reject(makeError("Analysis cancelled.", { kind: "aborted" }))
    );

    xhr.send(form);
  });

  return { promise, abort: () => xhr.abort() };
}

function makeError(message, props) {
  return Object.assign(new Error(message), props);
}
