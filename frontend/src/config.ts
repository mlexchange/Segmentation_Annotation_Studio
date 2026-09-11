/**
 * Base URL for the backend API.
 * - Default is derived from Vite's `BASE_URL` (itself driven by `VITE_BASE_PATH`
 *   at build time, see vite.config.ts): '/' when root-hosted (dev's `/api`
 *   proxy, or the production container serving the SPA same-origin), or the
 *   trimmed subpath (e.g. `/bl832/seg_studio`) when hosted behind a
 *   path-stripping reverse proxy, so `fetch(`${API_BASE}/api/...`)` still
 *   resolves to a path the proxy actually routes.
 * - Set `VITE_API_BASE` at build time only for split deployments where the API
 *   lives on a different origin (e.g. `https://api.example.com`).
 */
export const API_BASE =
  import.meta.env.VITE_API_BASE?.trim() || import.meta.env.BASE_URL.replace(/\/$/, '');

/**
 * URL of the user documentation site (MkDocs).
 * - Default targets the local `mkdocs serve` address used during development.
 * - Set `VITE_DOCS_URL` at build time to point at a deployed docs site
 *   (e.g. `https://docs.example.com`).
 */
export const DOCS_URL = import.meta.env.VITE_DOCS_URL?.trim() || 'http://127.0.0.1:8000';

/**
 * "Bugs & Feature Requests" Google Form. Bundled by default (the sidebar
 * "Feedback" button is always present via start_all.sh — no env needed); an env
 * var can override per-deployment.
 * - `VITE_FEEDBACK_FORM_URL`: the form's `…/viewform` URL.
 * - `VITE_FEEDBACK_ENTRY_ID`: the numeric `entry.<id>` of the single long-answer
 *   "context" question to prefill. When set, the app context is prefilled into it.
 */
export const FEEDBACK_FORM_URL =
  import.meta.env.VITE_FEEDBACK_FORM_URL?.trim() ||
  'https://docs.google.com/forms/d/e/1FAIpQLSdf3J6qTOohzPl65DvIlIKo49Yd3IEnLKFlEdhQy26vlwptew/viewform';
export const FEEDBACK_ENTRY_ID =
  import.meta.env.VITE_FEEDBACK_ENTRY_ID?.trim() ||
  '985606833'; // "System Context (Hidden)" paragraph field

/** App version + short git commit, injected at build time (see vite.config.ts). */
export const APP_VERSION: string = typeof __APP_VERSION__ === 'string' ? __APP_VERSION__ : 'dev';
export const GIT_COMMIT: string = typeof __GIT_COMMIT__ === 'string' ? __GIT_COMMIT__ : 'dev';
