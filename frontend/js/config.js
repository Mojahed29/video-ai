// Client-side constants. Kept in sync with the backend defaults so the UI can
// reject obviously-invalid files before uploading (saving the user a minute).
export const MAX_FILE_BYTES = 100 * 1024 * 1024; // 100 MB
export const MAX_DURATION_S = 60;
export const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".webm", ".mkv"];

export const ANALYZE_ENDPOINT = "/api/analyze";

// Score bands (likelihood the video is AI-generated).
export const BAND_LOW = 35; // below this: likely authentic
export const BAND_HIGH = 65; // above this: likely AI-generated

// Staged progress for the (monolithic) analysis request. Upload progress is
// real; the analysis stages below are time-estimated and advance on a timer,
// holding on the final stage until the server responds. They communicate
// *what the server is doing*, not a precise percentage.
export const ANALYSIS_STAGES = [
  { id: "extract", label: "Extracting frames", approxMs: 4000 },
  { id: "visual", label: "Scoring visual track", approxMs: 16000 },
  { id: "audio", label: "Scoring audio", approxMs: 12000 },
  { id: "fuse", label: "Fusing results", approxMs: 3000 },
];

export const UPLOAD_STAGE = { id: "upload", label: "Uploading video" };
