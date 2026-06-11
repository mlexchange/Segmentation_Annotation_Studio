/** In dev, empty base uses same origin so Vite's proxy applies. */
export const API_BASE =
  import.meta.env.VITE_API_BASE?.trim() ||
  (import.meta.env.DEV ? '' : 'http://127.0.0.1:8002');
