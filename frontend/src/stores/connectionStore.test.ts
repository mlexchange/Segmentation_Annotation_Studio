import { afterEach, describe, expect, it } from 'vitest';
import { useConnectionStore } from './connectionStore';

const INITIAL = useConnectionStore.getState();

afterEach(() => {
  useConnectionStore.setState(INITIAL, true);
});

describe('connectionStore', () => {
  it('starts with no connection and status unknown', () => {
    const s = useConnectionStore.getState();
    expect(s.kind).toBeNull();
    expect(s.status).toBe('unknown');
  });

  it('setConnection resets status to unknown', () => {
    useConnectionStore.getState().setStatus('ok');
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://example',
      label: 'Example',
      sampleCount: 3,
    });
    const s = useConnectionStore.getState();
    expect(s.kind).toBe('tiled');
    expect(s.status).toBe('unknown');
  });

  it('setStatus updates status independently of other fields', () => {
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://example',
      label: 'Example',
      sampleCount: 3,
    });
    useConnectionStore.getState().setStatus('error');
    const s = useConnectionStore.getState();
    expect(s.status).toBe('error');
    expect(s.serverUri).toBe('http://example');
  });

  it('clearConnection resets status to unknown along with everything else', () => {
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://example',
      label: 'Example',
      sampleCount: 3,
    });
    useConnectionStore.getState().setStatus('ok');
    useConnectionStore.getState().clearConnection();
    const s = useConnectionStore.getState();
    expect(s.kind).toBeNull();
    expect(s.status).toBe('unknown');
  });
});
