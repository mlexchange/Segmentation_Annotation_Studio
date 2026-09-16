/**
 * Persists the purely cosmetic display sliders (brightness/contrast/levels/
 * gamma/colormap/clahe/sharpen/blur) across reloads, scoped globally as a
 * viewer preference — not per-sample, not part of the draft. `denoise` is
 * deliberately excluded: it's a real training input tuned to one volume's
 * noise level, and carrying it over to a different volume would silently
 * mis-filter it (see datasetStore.ts's own note on why it resets).
 */
import type { ColormapName } from '@/lib/colormaps';

export interface DisplayPrefs {
  brightness: number;
  contrast: number;
  levelsLo: number;
  levelsHi: number;
  gamma: number;
  colormap: ColormapName;
  clahe: boolean;
  sharpen: boolean;
  blur: number;
}

const STORAGE_KEY = 'finch:displayPrefs';

export function loadDisplayPrefs(): Partial<DisplayPrefs> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function saveDisplayPrefs(prefs: DisplayPrefs): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // Storage unavailable (private browsing, quota, etc.) — cosmetic prefs
    // just won't persist this session; nothing to recover from.
  }
}
