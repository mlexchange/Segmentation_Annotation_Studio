import { afterEach, describe, expect, it } from 'vitest';
import { loadDisplayPrefs, saveDisplayPrefs, type DisplayPrefs } from './displayPrefs';

const FULL: DisplayPrefs = {
  brightness: 10,
  contrast: -5,
  levelsLo: 20,
  levelsHi: 230,
  gamma: 1.2,
  colormap: 'viridis',
  clahe: true,
  sharpen: false,
  blur: 1.5,
};

afterEach(() => {
  localStorage.clear();
});

describe('displayPrefs', () => {
  it('returns an empty object when nothing has been saved', () => {
    expect(loadDisplayPrefs()).toEqual({});
  });

  it('round-trips a saved prefs object', () => {
    saveDisplayPrefs(FULL);
    expect(loadDisplayPrefs()).toEqual(FULL);
  });

  it('returns an empty object for corrupt stored JSON instead of throwing', () => {
    localStorage.setItem('finch:displayPrefs', '{not json');
    expect(loadDisplayPrefs()).toEqual({});
  });

  it('returns an empty object when the stored value is not an object', () => {
    localStorage.setItem('finch:displayPrefs', '"just a string"');
    expect(loadDisplayPrefs()).toEqual({});
  });
});
