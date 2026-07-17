/**
 * samClient — main-thread singleton wrapping the SAM Web Worker.
 *
 * Owns one worker for the whole app: lazily spawns it on first use, tracks load
 * status, and turns the worker's message protocol into promises. The canvas
 * encodes the current slice once, then issues a decode per click.
 */
export type SamStatus = 'idle' | 'loading-model' | 'encoding' | 'ready' | 'unsupported';
export interface PromptPoint { x: number; y: number; label: 0 | 1 }
export interface PromptBox { x0: number; y0: number; x1: number; y1: number }
export type Granularity = 'auto' | 'fine' | 'medium' | 'coarse';
export interface SamMask { mask: Uint8Array; width: number; height: number; score: number }

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

class SamClient {
  private worker: Worker | null = null;
  private status: SamStatus = 'idle';
  private backend: string | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Set<(s: SamStatus) => void>();
  private initPromise: Promise<void> | null = null;

  /** Current worker status (model/encode lifecycle). */
  getStatus(): SamStatus { return this.status; }
  /** The compute backend in use ('webgpu' | 'wasm'), or null before load. */
  getBackend(): string | null { return this.backend; }

  /** Subscribe to status changes; fires immediately with the current status.
   *  Returns an unsubscribe function. */
  subscribe(fn: (s: SamStatus) => void): () => void {
    this.listeners.add(fn);
    fn(this.status);
    return () => { this.listeners.delete(fn); };
  }

  /** Update status and notify all subscribers. */
  private setStatus(s: SamStatus) {
    this.status = s;
    this.listeners.forEach((fn) => fn(s));
  }

  /** Spawn the worker and load the model (idempotent). */
  init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    this.initPromise = new Promise<void>((resolve, reject) => {
      try {
        this.worker = new Worker(new URL('./samWorker.ts', import.meta.url), { type: 'module' });
      } catch (err) {
        this.setStatus('unsupported');
        reject(err instanceof Error ? err : new Error(String(err)));
        return;
      }
      this.worker.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (msg.type === 'status') {
          this.backend = msg.backend ?? this.backend;
          this.setStatus(msg.status);
          if (msg.status === 'ready') resolve();
          if (msg.status === 'unsupported') reject(new Error(msg.message ?? 'unsupported'));
        } else if (msg.type === 'encoded' || msg.type === 'decoded') {
          this.pending.get(msg.id)?.resolve(msg);
          this.pending.delete(msg.id);
        } else if (msg.type === 'error') {
          this.pending.get(msg.id)?.reject(new Error(msg.message));
          this.pending.delete(msg.id);
        }
      };
      this.worker.onerror = (e) => {
        this.setStatus('unsupported');
        reject(new Error(e.message || 'SAM worker error'));
      };
      this.worker.postMessage({ type: 'init' });
    });
    // Allow a fresh init attempt after a failure (e.g. once the model is
    // vendored) without reloading the page: clear the cached rejected promise
    // and tear down the dead worker. The original rejection still reaches callers.
    this.initPromise.catch(() => {
      this.initPromise = null;
      try { this.worker?.terminate(); } catch { /* noop */ }
      this.worker = null;
    });
    return this.initPromise;
  }

  /** Post a tagged message to the worker and resolve when its matching reply
   *  (by auto-assigned id) arrives. `transfer` ownership-transfers buffers. */
  private request<T>(payload: Record<string, unknown>, transfer?: Transferable[]): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.worker!.postMessage({ ...payload, id }, transfer ?? []);
    });
  }

  /** Encode a slice image (any canvas source — bake brightness/contrast in to
   *  let SAM "see" what the user sees). Heavy; run once per slice/adjustment. */
  async encode(source: CanvasImageSource): Promise<void> {
    await this.init();
    this.setStatus('encoding');
    const bitmap = await createImageBitmap(source);
    try {
      await this.request<{ type: 'encoded' }>({ type: 'encode', bitmap }, [bitmap]);
    } finally {
      this.setStatus('ready');
    }
  }

  /** Decode a mask for the given prompts (image-normalised [0,1]). */
  async decode(
    points: PromptPoint[],
    box: PromptBox | null,
    granularity: Granularity,
    threshold: number,
  ): Promise<SamMask> {
    const res = await this.request<{ mask: ArrayBuffer; width: number; height: number; score: number }>(
      { type: 'decode', points, box, granularity, threshold },
    );
    return { mask: new Uint8Array(res.mask), width: res.width, height: res.height, score: res.score };
  }
}

export const samClient = new SamClient();

/** WebGPU is the fast path; absence isn't fatal (WASM fallback) but signals it. */
export function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}
