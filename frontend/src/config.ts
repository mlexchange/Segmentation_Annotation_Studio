/**
 * Base URL for the backend API.
 * - Default is '' (same origin): dev uses Vite's `/api` proxy; the production
 *   container serves the SPA from FastAPI itself, so `/api` is same-origin too.
 * - Set `VITE_API_BASE` at build time only for split deployments where the API
 *   lives on a different origin (e.g. `https://api.example.com`).
 */
export const API_BASE = import.meta.env.VITE_API_BASE?.trim() || '';

/**
 * URL of the user documentation site (MkDocs).
 * - Default targets the local `mkdocs serve` address used during development.
 * - Set `VITE_DOCS_URL` at build time to point at a deployed docs site
 *   (e.g. `https://docs.example.com`).
 */
export const DOCS_URL = import.meta.env.VITE_DOCS_URL?.trim() || 'http://127.0.0.1:8000';
