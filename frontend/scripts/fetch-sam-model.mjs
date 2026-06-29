#!/usr/bin/env node
/**
 * Vendor the SlimSAM model files for fully-offline use.
 *
 * By default the SAM magic tool streams the model from the Hugging Face CDN on
 * first use (and the browser caches it). On air-gapped lab machines that won't
 * work — run this once on a networked machine to download the model into
 * frontend/public/models/slimsam-77-uniform/, then set USE_LOCAL_MODEL = true
 * in src/lib/sam/samWorker.ts.
 *
 *   node scripts/fetch-sam-model.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'Xenova/slimsam-77-uniform';
const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models', 'slimsam-77-uniform');

async function main() {
  const treeUrl = `https://huggingface.co/api/models/${REPO}/tree/main?recursive=true`;
  const tree = await (await fetch(treeUrl)).json();
  const files = tree.filter((e) => e.type === 'file').map((e) => e.path);
  console.log(`Downloading ${files.length} files for ${REPO} → ${OUT}`);
  for (const path of files) {
    const url = `https://huggingface.co/${REPO}/resolve/main/${path}`;
    const res = await fetch(url);
    if (!res.ok) { console.warn(`  skip ${path} (${res.status})`); continue; }
    const buf = Buffer.from(await res.arrayBuffer());
    const dest = join(OUT, path);
    await mkdir(dirname(dest), { recursive: true });
    await writeFile(dest, buf);
    console.log(`  ${path} (${(buf.length / 1e6).toFixed(1)} MB)`);
  }
  console.log('Done. Set USE_LOCAL_MODEL = true in src/lib/sam/samWorker.ts.');
}

main().catch((e) => { console.error(e); process.exit(1); });
