import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * samClient exports only a module-level singleton (no class, no reset), so
 * each test re-imports the module fresh via vi.resetModules() to get an
 * un-contaminated instance — otherwise `init()`'s cached promise/worker would
 * leak state across tests.
 */
class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  terminated = false;
  posted: any[] = [];

  constructor(_url: unknown, _opts?: unknown) {
    FakeWorker.instances.push(this);
  }

  postMessage(msg: any) {
    this.posted.push(msg);
  }

  terminate() {
    this.terminated = true;
  }

  // Test helper: simulate the worker sending a message back.
  emit(data: any) {
    this.onmessage?.({ data } as MessageEvent);
  }
}

beforeEach(() => {
  FakeWorker.instances = [];
  vi.stubGlobal('Worker', FakeWorker as any);
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ close: () => {} }) as any));
  vi.resetModules();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function freshClient() {
  const mod = await import('./samClient');
  return mod;
}

describe('samClient.init', () => {
  it('starts idle before init', async () => {
    const { samClient } = await freshClient();
    expect(samClient.getStatus()).toBe('idle');
  });

  it('spawns exactly one worker across repeated init() calls (idempotent)', async () => {
    const { samClient } = await freshClient();
    const p1 = samClient.init();
    const p2 = samClient.init();
    expect(p1).toBe(p2);
    expect(FakeWorker.instances).toHaveLength(1);
  });

  it('resolves once the worker reports status "ready", and notifies subscribers', async () => {
    const { samClient } = await freshClient();
    const seen: string[] = [];
    samClient.subscribe((s) => seen.push(s));
    const p = samClient.init();
    FakeWorker.instances[0].emit({ type: 'status', status: 'loading-model' });
    FakeWorker.instances[0].emit({ type: 'status', status: 'ready', backend: 'wasm' });
    await expect(p).resolves.toBeUndefined();
    expect(samClient.getStatus()).toBe('ready');
    expect(samClient.getBackend()).toBe('wasm');
    expect(seen).toContain('loading-model');
    expect(seen).toContain('ready');
  });

  it('rejects when the worker reports "unsupported", and allows retrying init after', async () => {
    const { samClient } = await freshClient();
    const p = samClient.init();
    FakeWorker.instances[0].emit({ type: 'status', status: 'unsupported', message: 'no webgpu/wasm' });
    await expect(p).rejects.toThrow('no webgpu/wasm');
    expect(samClient.getStatus()).toBe('unsupported');

    // A later init() attempt spawns a fresh worker rather than reusing the dead one.
    await Promise.resolve(); // let the internal .catch() teardown run
    const p2 = samClient.init();
    expect(FakeWorker.instances.length).toBeGreaterThanOrEqual(2);
    FakeWorker.instances[FakeWorker.instances.length - 1].emit({ type: 'status', status: 'ready' });
    await expect(p2).resolves.toBeUndefined();
  });

  it('rejects and sets "unsupported" when the Worker constructor throws', async () => {
    vi.stubGlobal('Worker', class {
      constructor() { throw new Error('workers disabled'); }
    } as any);
    vi.resetModules();
    const { samClient } = await freshClient();
    await expect(samClient.init()).rejects.toThrow('workers disabled');
    expect(samClient.getStatus()).toBe('unsupported');
  });

  it('rejects on a worker onerror event', async () => {
    const { samClient } = await freshClient();
    const p = samClient.init();
    FakeWorker.instances[0].onerror?.({ message: 'boom' });
    await expect(p).rejects.toThrow('boom');
    expect(samClient.getStatus()).toBe('unsupported');
  });
});

describe('samClient.encode/decode', () => {
  it('encode() initializes, sets status to encoding then back to ready', async () => {
    const { samClient } = await freshClient();
    const statuses: string[] = [];
    samClient.subscribe((s) => statuses.push(s));

    const encodePromise = samClient.encode({} as any);
    FakeWorker.instances[0].emit({ type: 'status', status: 'ready' });
    // Let init() resolve and encode() proceed to post the encode message.
    await Promise.resolve();
    await Promise.resolve();
    const encodeMsg = FakeWorker.instances[0].posted.find((m) => m.type === 'encode');
    expect(encodeMsg).toBeTruthy();
    FakeWorker.instances[0].emit({ type: 'encoded', id: encodeMsg.id });
    await encodePromise;

    expect(statuses).toContain('encoding');
    expect(samClient.getStatus()).toBe('ready');
  });

  it('decode() resolves a SamMask from the worker reply', async () => {
    const { samClient } = await freshClient();
    const p = samClient.init();
    FakeWorker.instances[0].emit({ type: 'status', status: 'ready' });
    await p;

    const decodePromise = samClient.decode(
      [{ x: 1, y: 2, label: 1 }], null, 'auto', 0.5,
    );
    const decodeMsg = FakeWorker.instances[0].posted.find((m) => m.type === 'decode');
    expect(decodeMsg).toBeTruthy();
    const buf = new Uint8Array([1, 0, 1, 0]).buffer;
    FakeWorker.instances[0].emit({
      type: 'decoded', id: decodeMsg.id, mask: buf, width: 2, height: 2, score: 0.9,
    });
    const result = await decodePromise;
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    expect(result.score).toBe(0.9);
    expect([...result.mask]).toEqual([1, 0, 1, 0]);
  });

  it('rejects a pending request when the worker reports an error for its id', async () => {
    const { samClient } = await freshClient();
    const p = samClient.init();
    FakeWorker.instances[0].emit({ type: 'status', status: 'ready' });
    await p;

    const decodePromise = samClient.decode([], null, 'auto', 0.5);
    const decodeMsg = FakeWorker.instances[0].posted.find((m) => m.type === 'decode');
    FakeWorker.instances[0].emit({ type: 'error', id: decodeMsg.id, message: 'decode failed' });
    await expect(decodePromise).rejects.toThrow('decode failed');
  });
});

describe('webgpuAvailable', () => {
  it('reflects navigator.gpu presence', async () => {
    const { webgpuAvailable } = await freshClient();
    const original = (navigator as any).gpu;
    (navigator as any).gpu = {};
    expect(webgpuAvailable()).toBe(true);
    delete (navigator as any).gpu;
    expect(webgpuAvailable()).toBe(false);
    if (original !== undefined) (navigator as any).gpu = original;
  });
});
