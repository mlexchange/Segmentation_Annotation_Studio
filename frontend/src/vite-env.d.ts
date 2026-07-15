/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_DOCS_URL?: string;
  readonly VITE_FEEDBACK_FORM_URL?: string;
  readonly VITE_FEEDBACK_ENTRY_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** Build-time constants injected via vite.config.ts `define`. */
declare const __APP_VERSION__: string;
declare const __GIT_COMMIT__: string;
