// `vitest/config` re-exports Vite's defineConfig with the `test` option typed.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
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
