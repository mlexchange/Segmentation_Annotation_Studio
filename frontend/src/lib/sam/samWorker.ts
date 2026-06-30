/**
 * SAM Web Worker — runs Segment Anything entirely in the browser via
 * Transformers.js (onnxruntime-web). Kept off the UI thread because the image
 * encoder is heavy (~1–4 s on WebGPU, slower on WASM).
 *
 * Lifecycle:
 *   init   → load model + processor (WebGPU, falling back to WASM). Posts the
 *            backend it ended up on, or 'unsupported' if nothing loads.
 *   encode → run the vision encoder ONCE for the current slice; cache the
 *            embedding + the processor's reshaped/original sizes + the source
 *            pixels (so a WASM fallback can re-encode without a round-trip).
 *   decode → run the lightweight prompt decoder for point and/or box prompts
 *            and return the selected mask as a flat Uint8Array.
 *
 * Prompts arrive normalised to [0,1] (x/imageW, y/imageH); the worker scales
 * them to the processor's reshaped input size. Positive points grow the object,
 * negative points (label 0) carve regions out, and a box constrains the search
 * to a region (the strongest quality lever on low-contrast scans). SAM returns
 * 3 masks per prompt at increasing scale; `granularity` selects which.
 *
 * Robustness: WebGPU can load fine yet throw at session-run time, and the
 * processor mutates its RawImage in place — so we keep the raw pixels (not the
 * RawImage) and rebuild a fresh one per attempt, retrying once on WASM.
 */
import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
  env,
  type PreTrainedModel,
  type Processor,
} from '@huggingface/transformers';

// SlimSAM-77 (uniform) — smallest turnkey SAM. Served from the HF CDN by
// default (browser-cached after first load). To run fully offline, vendor the
// files under frontend/public/models/ and flip USE_LOCAL_MODEL (see
// frontend/scripts/fetch-sam-model.mjs).
const USE_LOCAL_MODEL = false;
const MODEL_ID = USE_LOCAL_MODEL ? 'slimsam-77-uniform' : 'Xenova/slimsam-77-uniform';

if (USE_LOCAL_MODEL) {
  env.allowRemoteModels = false;
  env.localModelPath = '/models/';
}
if (env.backends?.onnx?.wasm) env.backends.onnx.wasm.numThreads = 1;

interface PromptPoint { x: number; y: number; label: 0 | 1 }
type Box = { x0: number; y0: number; x1: number; y1: number };
type Granularity = 'auto' | 'fine' | 'medium' | 'coarse';
type Device = 'webgpu' | 'wasm';

let model: PreTrainedModel | null = null;
let processor: Processor | null = null;
let backend: Device = 'wasm';
// Cached per-slice encoder state + the source pixels (kept as raw RGBA so we
// can rebuild a fresh RawImage per attempt — the processor mutates it).
let lastPixels: Uint8ClampedArray | null = null;
let lastW = 0;
let lastH = 0;
let imageEmbeddings: Record<string, Tensor> | null = null;
let reshaped: [number, number] | null = null; // [height, width] after processor resize
let originalSizes: [number, number][] | null = null;

type InMsg =
  | { type: 'init' }
  | { type: 'encode'; id: number; bitmap: ImageBitmap }
  | { type: 'decode'; id: number; points: PromptPoint[]; box: Box | null; granularity: Granularity; threshold: number };

/** Post a message back to the main thread, optionally transferring buffers. */
function post(msg: unknown, transfer?: Transferable[]) {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

/** Load the SAM model + processor onto a specific device and record the backend. */
async function loadOn(device: Device) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model = await SamModel.from_pretrained(MODEL_ID, { device } as any);
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  backend = device;
}

/** Load on the best available device, trying WebGPU first then WASM; returns the
 *  device that succeeded or throws if none load. */
async function load(): Promise<Device> {
  const hasGpu = typeof (navigator as Navigator & { gpu?: unknown }).gpu !== 'undefined';
  const devices: Device[] = hasGpu ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr: unknown;
  for (const device of devices) {
    try {
      await loadOn(device);
      return device;
    } catch (err) {
      lastErr = err;
      console.error(`[SAM] load on ${device} failed`, err);
    }
  }
  throw lastErr ?? new Error('SAM model failed to load');
}

/** Reload the model on WASM after a WebGPU runtime failure and re-announce ready. */
async function fallbackToWasm(reason: unknown) {
  console.warn('[SAM] WebGPU runtime failure — falling back to WASM', reason);
  await loadOn('wasm');
  post({ type: 'status', status: 'ready', backend: 'wasm' });
}

/** Fresh RawImage from the cached pixels (preprocessing mutates it). */
function freshRaw(): RawImage {
  return new RawImage(new Uint8ClampedArray(lastPixels!), lastW, lastH, 4).rgb();
}

/** Run the vision encoder on the cached pixels; caches the embedding plus the
 *  processor's reshaped and original sizes for later decode calls. */
async function runEncoder() {
  const inputs = await processor!(freshRaw());
  imageEmbeddings = await (model as PreTrainedModel & {
    get_image_embeddings: (i: unknown) => Promise<Record<string, Tensor>>;
  }).get_image_embeddings(inputs);
  const r = (inputs as { reshaped_input_sizes: number[][] }).reshaped_input_sizes[0];
  reshaped = [r[0], r[1]];
  originalSizes = (inputs as { original_sizes: [number, number][] }).original_sizes;
}

/** Cache the bitmap's RGBA pixels and run the encoder, retrying once on WASM if
 *  a WebGPU run fails. Heavy; called once per slice. */
async function encode(bitmap: ImageBitmap) {
  if (!model || !processor) throw new Error('model not loaded');
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  lastW = bitmap.width;
  lastH = bitmap.height;
  lastPixels = new Uint8ClampedArray(data);
  bitmap.close();

  try {
    await runEncoder();
  } catch (err) {
    if (backend === 'webgpu') {
      await fallbackToWasm(err);
      await runEncoder();
    } else {
      throw err;
    }
  }
}

/** Run the prompt decoder on the cached embedding for the given normalised
 *  points/box, threshold the masks, pick a channel per `granularity` (auto =
 *  highest IoU; else by area), and return the chosen mask as a flat Uint8Array. */
async function runDecoder(points: PromptPoint[], box: Box | null, granularity: Granularity, threshold: number) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const inputs: any = { ...imageEmbeddings };
  const [rh, rw] = reshaped!;

  // SamModel.forward unconditionally dereferences input_points.dims, so we must
  // always pass points. For a box with no clicks, send one padding point
  // (label -10 = ignored by the prompt encoder) to satisfy that.
  const flatPts: number[] = [];
  const flatLabels: bigint[] = [];
  for (const p of points) {
    flatPts.push(p.x * rw, p.y * rh);
    flatLabels.push(BigInt(p.label));
  }
  if (flatLabels.length === 0) {
    flatPts.push(0, 0);
    flatLabels.push(BigInt(-10));
  }
  inputs.input_points = new Tensor('float32', flatPts, [1, 1, flatLabels.length, 2]);
  inputs.input_labels = new Tensor('int64', flatLabels, [1, 1, flatLabels.length]);
  if (box) {
    inputs.input_boxes = new Tensor(
      'float32',
      [box.x0 * rw, box.y0 * rh, box.x1 * rw, box.y1 * rh],
      [1, 1, 4],
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const outputs: any = await (model as PreTrainedModel)(inputs);

  const masks = await (processor as Processor & {
    post_process_masks: (m: unknown, o: unknown, r: unknown, opts: unknown) => Promise<Tensor[]>;
  }).post_process_masks(outputs.pred_masks, originalSizes, [reshaped], { mask_threshold: threshold });

  const maskTensor = masks[0]; // dims [1, numMasks, H, W], bool
  const [, numMasks, H, W] = maskTensor.dims as number[];
  const iou = outputs.iou_scores.data as Float32Array; // length numMasks
  const src = maskTensor.data as Uint8Array;

  // Pick the mask channel. 'auto' → highest IoU. Otherwise order the channels by
  // area (fine = smallest selection … coarse = largest) and pick accordingly.
  let chosen = 0;
  if (granularity === 'auto' || numMasks === 1) {
    for (let i = 1; i < numMasks; i++) if (iou[i] > iou[chosen]) chosen = i;
  } else {
    const areas: Array<{ i: number; a: number }> = [];
    for (let i = 0; i < numMasks; i++) {
      let a = 0;
      const off = i * H * W;
      for (let j = 0; j < H * W; j++) a += src[off + j];
      areas.push({ i, a });
    }
    areas.sort((p, q) => p.a - q.a); // ascending area
    const idx = granularity === 'fine' ? 0 : granularity === 'coarse' ? areas.length - 1 : (areas.length >> 1);
    chosen = areas[idx].i;
  }

  const out = new Uint8Array(H * W);
  const offset = chosen * H * W;
  for (let i = 0; i < H * W; i++) out[i] = src[offset + i] ? 1 : 0;
  return { mask: out, width: W, height: H, score: iou[chosen] };
}

/** Decode a mask for the current slice's embedding, retrying once on WASM
 *  (re-encoding first) if a WebGPU run fails. Requires a prior encode. */
async function decode(points: PromptPoint[], box: Box | null, granularity: Granularity, threshold: number) {
  if (!model || !processor || !imageEmbeddings || !reshaped || !originalSizes) {
    throw new Error('encode must run before decode');
  }
  try {
    return await runDecoder(points, box, granularity, threshold);
  } catch (err) {
    if (backend === 'webgpu' && lastPixels) {
      await fallbackToWasm(err);
      await runEncoder();
      return await runDecoder(points, box, granularity, threshold);
    }
    throw err;
  }
}

/** Dispatch init/encode/decode messages from the main thread and post back the
 *  result (or an error/status reply tagged with the request id). */
self.onmessage = async (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      post({ type: 'status', status: 'loading-model' });
      const device = await load();
      post({ type: 'status', status: 'ready', backend: device });
    } else if (msg.type === 'encode') {
      await encode(msg.bitmap);
      post({ type: 'encoded', id: msg.id });
    } else if (msg.type === 'decode') {
      const { mask, width, height, score } = await decode(msg.points, msg.box, msg.granularity, msg.threshold);
      post({ type: 'decoded', id: msg.id, mask: mask.buffer, width, height, score }, [mask.buffer]);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error('[SAM] worker error', err);
    if (msg.type === 'init') post({ type: 'status', status: 'unsupported', message });
    else post({ type: 'error', id: (msg as { id: number }).id, message });
  }
};
