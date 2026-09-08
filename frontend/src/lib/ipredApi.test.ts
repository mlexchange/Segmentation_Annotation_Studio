import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  getIpredComposition, getIpredSetup, ipredChannelUrl, ipredHealth, ipredInfer,
  ipredPreprocess, ipredRunCommitUrl, ipredRunProbaUrl, ipredRunStatusUrl,
  ipredThresholdClass, ipredTrain, listIpredCompositions, listIpredModules,
  listIpredSetups, listIpredTrainers, openIpredSession, previewIpredComposition,
  upsertIpredComposition, upsertIpredSetup,
} from './ipredApi';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function ok(body: unknown) {
  return { ok: true, json: async () => body };
}

function errRes(status: number, body: string, statusText = 'Error') {
  return { ok: false, status, statusText, text: async () => body };
}

describe('ipredHealth', () => {
  it('returns the parsed health body', async () => {
    (fetch as any).mockResolvedValue(ok({ status: 'ok' }));
    expect(await ipredHealth()).toEqual({ status: 'ok' });
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/ipred/health'));
  });

  it('throws a parsed error with the detail string on failure', async () => {
    (fetch as any).mockResolvedValue(errRes(503, JSON.stringify({ detail: 'ipred unreachable' })));
    await expect(ipredHealth()).rejects.toThrow('ipred unreachable');
  });

  it('falls back to raw text when the error body is not JSON', async () => {
    (fetch as any).mockResolvedValue(errRes(500, 'plain text failure'));
    await expect(ipredHealth()).rejects.toThrow('plain text failure');
  });

  it('falls back to statusText when the error body is empty', async () => {
    (fetch as any).mockResolvedValue(errRes(500, '', 'Internal Server Error'));
    await expect(ipredHealth()).rejects.toThrow('Internal Server Error');
  });
});

describe('openIpredSession', () => {
  it('POSTs the payload and returns the session', async () => {
    (fetch as any).mockResolvedValue(ok({ session_id: 's1', project_id: 'p1' }));
    const result = await openIpredSession({ kind: 'local', source: 'x.tif' });
    expect(result).toEqual({ session_id: 's1', project_id: 'p1' });
    const [url, init] = (fetch as any).mock.calls[0];
    expect(url).toContain('/api/ipred/sessions');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ kind: 'local', source: 'x.tif' });
  });
});

describe('listIpredSetups', () => {
  it('returns the setups array', async () => {
    (fetch as any).mockResolvedValue(ok({ setups: [{ id: 's1' }] }));
    expect(await listIpredSetups()).toEqual([{ id: 's1' }]);
  });

  it('defaults to an empty array when the setups key is missing', async () => {
    (fetch as any).mockResolvedValue(ok({}));
    expect(await listIpredSetups()).toEqual([]);
  });
});

describe('getIpredSetup', () => {
  it('encodes the setup id in the URL', async () => {
    (fetch as any).mockResolvedValue(ok({ id: 'a/b' }));
    await getIpredSetup('a/b');
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining(encodeURIComponent('a/b')));
  });
});

describe('upsertIpredSetup', () => {
  it('POSTs the setup payload', async () => {
    (fetch as any).mockResolvedValue(ok({ id: 's1' }));
    await upsertIpredSetup({ name: 'setup', kind: 'procedure' });
    const [, init] = (fetch as any).mock.calls[0];
    expect(init.method).toBe('POST');
  });
});

describe('listIpredTrainers', () => {
  it('returns the trainers array, defaulting to empty', async () => {
    (fetch as any).mockResolvedValue(ok({}));
    expect(await listIpredTrainers()).toEqual([]);
    (fetch as any).mockResolvedValue(ok({ trainers: ['catboost'] }));
    expect(await listIpredTrainers()).toEqual(['catboost']);
  });
});

describe('listIpredModules', () => {
  it('returns the modules array, defaulting to empty', async () => {
    (fetch as any).mockResolvedValue(ok({}));
    expect(await listIpredModules()).toEqual([]);
  });
});

describe('listIpredCompositions / getIpredComposition / upsertIpredComposition', () => {
  it('lists compositions, defaulting to empty', async () => {
    (fetch as any).mockResolvedValue(ok({}));
    expect(await listIpredCompositions()).toEqual([]);
  });

  it('gets one composition by id', async () => {
    (fetch as any).mockResolvedValue(ok({ id: 'c1', name: 'comp', nodes: [], outputs: [] }));
    const result = await getIpredComposition('c1');
    expect(result.id).toBe('c1');
  });

  it('upserts a composition via POST', async () => {
    (fetch as any).mockResolvedValue(ok({ id: 'c2', name: 'x', nodes: [], outputs: [] }));
    await upsertIpredComposition({ name: 'x', nodes: [], outputs: [] });
    const [url, init] = (fetch as any).mock.calls[0];
    expect(url).toContain('/api/ipred/compositions');
    expect(init.method).toBe('POST');
  });
});

describe('previewIpredComposition', () => {
  it('defaults the name to "preview" when omitted', async () => {
    (fetch as any).mockResolvedValue(ok({ preview_labels: [] }));
    await previewIpredComposition({ nodes: [], outputs: [] });
    const [, init] = (fetch as any).mock.calls[0];
    expect(JSON.parse(init.body).name).toBe('preview');
  });

  it('keeps an explicit name', async () => {
    (fetch as any).mockResolvedValue(ok({ preview_labels: [] }));
    await previewIpredComposition({ name: 'draft', nodes: [], outputs: [] });
    const [, init] = (fetch as any).mock.calls[0];
    expect(JSON.parse(init.body).name).toBe('draft');
  });
});

describe('ipredPreprocess', () => {
  it('POSTs and returns the preprocess result', async () => {
    (fetch as any).mockResolvedValue(ok({
      feature_id: 'f1', project_id: 'p1', setup_id: 's1', slice_index: 0,
      n_channels: 3, height: 64, width: 64, labels: [], cache_hit: false,
    }));
    const result = await ipredPreprocess({ session_id: 's1' });
    expect(result.feature_id).toBe('f1');
  });
});

describe('URL builders', () => {
  it('ipredChannelUrl encodes the feature id', () => {
    expect(ipredChannelUrl('feat/1', 2)).toContain(encodeURIComponent('feat/1'));
    expect(ipredChannelUrl('feat1', 2)).toContain('/channels/2');
  });

  it('ipredRunCommitUrl/StatusUrl/ProbaUrl build the expected paths', () => {
    expect(ipredRunCommitUrl('run/1')).toContain(`${encodeURIComponent('run/1')}/commit.png`);
    expect(ipredRunStatusUrl('run1')).toContain('run1/status.png');
    expect(ipredRunProbaUrl('run1', 3)).toContain('run1/proba/3.png');
  });
});

describe('ipredTrain / ipredInfer', () => {
  it('ipredTrain POSTs the payload and returns the result', async () => {
    (fetch as any).mockResolvedValue(ok({
      model_id: 'm1', feature_id: 'f1', trainer_id: 'catboost', class_ids: [1],
      train_accuracy: 0.9, n_train: 10, n_cal: 5, n_samples: 15, params: {},
    }));
    const result = await ipredTrain({ session_id: 's1', shapes: [] });
    expect(result.model_id).toBe('m1');
  });

  it('ipredInfer POSTs the payload and returns the result', async () => {
    (fetch as any).mockResolvedValue(ok({
      run_id: 'r1', model_id: 'm1', feature_id: 'f1', alpha: 0.05, class_ids: [1],
      counts: { singleton: 1, multi: 0, abstain: 0 },
    }));
    const result = await ipredInfer({ session_id: 's1' });
    expect(result.run_id).toBe('r1');
  });
});

describe('ipredThresholdClass', () => {
  it('POSTs to the run-scoped threshold-class endpoint', async () => {
    (fetch as any).mockResolvedValue(ok({
      run_id: 'r1', class_id: 1, class_index: 0, threshold: 0.5,
      width: 10, height: 10, n_positive: 5, label_map_b64: 'abc',
    }));
    const result = await ipredThresholdClass('r1', { class_id: 1, threshold: 0.5 });
    expect(result.n_positive).toBe(5);
    const [url] = (fetch as any).mock.calls[0];
    expect(url).toContain('/runs/r1/threshold-class');
  });
});
