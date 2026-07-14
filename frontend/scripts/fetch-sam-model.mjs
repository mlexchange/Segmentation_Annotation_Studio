#!/usr/bin/env node
/**
 * Vendor the SlimSAM model files for local/offline use of the Magic tool's
 * "Smart (AI)" (SAM) engine.
 *
 * The browser cannot stream the model from the Hugging Face CDN on many machines
 * (403 Forbidden via the Xet CDN), which greys out "Smart (AI)". This script
 * downloads the model into frontend/public/models/slimsam-77-uniform/ so the app
 * loads it locally (samWorker.ts is local-first with a remote fallback).
 *
 * Plain HTTP fetch also 403s, so we vendor via Python `huggingface_hub`
 * (snapshot_download). It uses the repo venv python: set PYTHON to override,
 * else falls back to ../.venv/bin/python then `python3`.
 *
 *   node scripts/fetch-sam-model.mjs
 *
 * start_all.sh runs this automatically (best-effort, backgrounded) on startup.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'Xenova/slimsam-77-uniform';
const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'public', 'models', 'slimsam-77-uniform');

/** Resolve a python interpreter: $PYTHON, then repo ../.venv, then python3. */
function resolvePython() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = join(HERE, '..', '..', '.venv', 'bin', 'python');
  if (existsSync(venv)) return venv;
  return 'python3';
}

const python = resolvePython();
console.log(`Vendoring ${REPO} → ${OUT}\n  using python: ${python}`);

// Ensure huggingface_hub is available (no-op if already installed).
spawnSync(python, ['-m', 'pip', 'install', '-q', 'huggingface_hub'], { stdio: 'inherit' });

const code = `
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(${JSON.stringify(REPO)}, local_dir=${JSON.stringify(OUT)})
print("SAM model vendored to", p)
`;
const res = spawnSync(python, ['-c', code], { stdio: 'inherit' });

if (res.status !== 0) {
  console.error('\nFetch failed. Ensure the machine has network access to huggingface.co');
  console.error('and that huggingface_hub is installed in the venv. Smart (AI) will fall');
  console.error('back to the remote model until the files are vendored.');
  process.exit(res.status ?? 1);
}
console.log('Done. Hard-refresh the app to load SAM from /models/.');
