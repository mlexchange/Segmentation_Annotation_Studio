#!/usr/bin/env node
/**
 * Vendor the SlimSAM model files for offline / CORS-safe use.
 *
 * Hugging Face's ONNX CDN (Xet) often returns 403 for bare browser/curl
 * downloads, which greys out Smart (AI). This script uses huggingface_hub
 * (Python) to pull weights into frontend/public/models/slimsam-77-uniform/.
 *
 *   uv pip install --python .venv/bin/python huggingface_hub   # once
 *   node frontend/scripts/fetch-sam-model.mjs
 *
 * Ensure USE_LOCAL_MODEL = true in src/lib/sam/samWorker.ts (default).
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'public', 'models', 'slimsam-77-uniform');
const REPO = 'Xenova/slimsam-77-uniform';

function pythonWithHub() {
  const candidates = [
    join(ROOT, '..', '.venv', 'bin', 'python'),
    'python3',
    'python',
  ];
  for (const p of candidates) {
    if ((p.includes('/') || p.startsWith('.')) && !existsSync(p)) continue;
    const r = spawnSync(p, ['-c', 'import huggingface_hub'], { encoding: 'utf8' });
    if (r.status === 0) return p;
  }
  return null;
}

const python = pythonWithHub();
if (!python) {
  console.error('Need Python with huggingface_hub. Try:');
  console.error('  uv pip install --python .venv/bin/python huggingface_hub');
  process.exit(1);
}

const script = `
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

out = Path(${JSON.stringify(OUT)})
out.mkdir(parents=True, exist_ok=True)
path = snapshot_download(${JSON.stringify(REPO)}, local_dir=str(out))
cache = out / ".cache"
if cache.exists():
    shutil.rmtree(cache)
print("Downloaded", ${JSON.stringify(REPO)}, "→", path)
for p in sorted(Path(path).rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(path)} {(p.stat().st_size / 1e6):.1f} MB")
`;

const result = spawnSync(python, ['-c', script], { stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('Done. USE_LOCAL_MODEL should be true in src/lib/sam/samWorker.ts.');
