import { describe, expect, it } from 'vitest';
import { canContinueFineTuning, fineTuningBlockedReason } from './runCompatibility';
import type { AnnotationClass } from '@/stores/classStore';

const cls = (...labels: string[]): AnnotationClass[] =>
  labels.map((label, i) => ({ classId: i + 1, label, color: '#ff0000', isVisible: true }));

const runCls = (...labels: string[]) => labels.map((label) => ({ label }));

describe('canContinueFineTuning', () => {
  it('accepts identical class lists', () => {
    expect(canContinueFineTuning(runCls('background', 'leaf'), cls('background', 'leaf'))).toBe(true);
  });

  it('ignores case and surrounding whitespace, matching the backend', () => {
    expect(canContinueFineTuning(runCls('Background', ' Leaf '), cls('background', 'leaf'))).toBe(true);
  });

  it('rejects a different class count', () => {
    expect(canContinueFineTuning(runCls('blob'), cls('background', 'leaf'))).toBe(false);
  });

  it('rejects reordered classes', () => {
    // The saved head's channel c means the run's class c, so order is load-bearing.
    expect(canContinueFineTuning(runCls('background', 'leaf'), cls('leaf', 'background'))).toBe(false);
  });

  it('rejects a rename at the same count', () => {
    // The dangerous case: nothing downstream would notice on its own.
    expect(canContinueFineTuning(runCls('blob'), cls('leaf'))).toBe(false);
  });

  it('treats two empty lists as compatible', () => {
    expect(canContinueFineTuning([], [])).toBe(true);
  });
});

describe('fineTuningBlockedReason', () => {
  it('returns null when the run can be continued', () => {
    expect(fineTuningBlockedReason(runCls('leaf'), cls('leaf'))).toBeNull();
  });

  it('names both class lists on a count mismatch, so the user can act on it', () => {
    const reason = fineTuningBlockedReason(runCls('blob'), cls('background', 'leaf'));
    expect(reason).toContain('blob');
    expect(reason).toContain('background, leaf');
  });

  it('names both class lists on a same-count mismatch', () => {
    const reason = fineTuningBlockedReason(runCls('blob'), cls('leaf'));
    expect(reason).toContain('blob');
    expect(reason).toContain('leaf');
  });

  it('points the user at applying instead, which does work across taxonomies', () => {
    const reason = fineTuningBlockedReason(runCls('blob'), cls('leaf'));
    expect(reason).toMatch(/apply it instead/i);
  });
});
