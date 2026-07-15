// `vitest/config` re-exports Vite's defineConfig with the `test` option typed.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';
import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

/** App version (package.json) + short git commit, injected at build time. */
function appVersion(): string {
  try { return JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')).version || 'dev'; }
  catch { return 'dev'; }
}
function gitCommit(): string {
  try { return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() || 'dev'; }
  catch { return 'dev'; }
}

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  define: {
    __APP_VERSION__: JSON.stringify(appVersion()),
    __GIT_COMMIT__: JSON.stringify(gitCommit()),
  },
  // Transformers.js bundles onnxruntime-web (wasm/webgpu); let it load its own
  // assets at runtime rather than having Vite pre-bundle/optimize it.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  // The SAM worker is an ES module worker (`new Worker(url, { type: 'module' })`).
  worker: { format: 'es' },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      // Backend origin — start_all.sh sets API_PROXY_TARGET to the port the
      // backend actually bound (it may fall back off 8002 if that port is busy).
      '/api': process.env.API_PROXY_TARGET || 'http://127.0.0.1:8002',
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
});
