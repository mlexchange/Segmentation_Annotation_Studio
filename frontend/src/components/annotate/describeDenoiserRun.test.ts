/**
 * `describeDenoiserRun` labels a saved denoiser run in the picker.
 *
 * Two runs used to be indistinguishable there unless their training scheme
 * differed. With two architectures and a tunable bottleneck that got worse, so
 * the label has to carry enough to tell them apart — and it reads an untyped
 * `Record<string, unknown>` straight off the backend, so it must degrade
 * gracefully rather than render "undefined".
 */
import { describe, expect, it } from 'vitest';
import { describeDenoiserRun } from './LearnedDenoiserPanel';

const run = (model_config: Record<string, unknown>) => ({ model_config });

describe('describeDenoiserRun', () => {
  it('names each known scheme', () => {
    expect(describeDenoiserRun(run({ training_scheme: 'n2n' }))).toBe('Noise2Noise');
    expect(describeDenoiserRun(run({ training_scheme: 'n2v' }))).toBe('Noise2Void');
    expect(describeDenoiserRun(run({ training_scheme: 'ae' }))).toBe('Autoencoder (bottleneck)');
  });

  it('appends the bottleneck so two autoencoder runs are distinguishable', () => {
    expect(describeDenoiserRun(run({ training_scheme: 'ae', ae_compression: 4 })))
      .toBe('Autoencoder (bottleneck) 4x');
    expect(describeDenoiserRun(run({ training_scheme: 'ae', ae_compression: 64 })))
      .toBe('Autoencoder (bottleneck) 64x');
  });

  it('omits the bottleneck for schemes that have none', () => {
    expect(describeDenoiserRun(run({ training_scheme: 'n2v' }))).not.toMatch(/x$/);
  });

  it('falls back to the raw scheme string for an unknown scheme', () => {
    // A run trained by a newer build must still be selectable, not blank.
    expect(describeDenoiserRun(run({ training_scheme: 'dae' }))).toBe('dae');
  });

  it('survives a run with no scheme recorded at all', () => {
    expect(describeDenoiserRun(run({}))).toBe('Denoiser');
  });
});
