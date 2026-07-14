/**
 * Annotation class color palettes.
 *
 * Two palettes are available:
 *  - DEFAULT_COLORS: the standard matplotlib tab20-style palette.
 *  - COLORBLIND_COLORS: a colorblind-safe palette (Okabe–Ito, extended with
 *    Paul Tol's muted set) for annotators with color vision deficiency.
 *
 * The active palette is chosen by the `colorblindMode` preference in
 * settingsStore (persisted to localStorage). Call `getClassPalette()` wherever
 * a new class color is auto-assigned so the choice is honored everywhere.
 */
import { useSettingsStore } from '@/stores/settingsStore';

/** Default color palette (matplotlib tab20 style). */
export const DEFAULT_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
];

/**
 * Colorblind-safe palette. The first eight entries are the Okabe–Ito palette
 * (with grey substituted for black so masks stay visible over dark images);
 * the remainder extend it with Paul Tol's muted colors. Ordered so adjacent
 * classes stay maximally distinguishable across deuteranopia, protanopia, and
 * tritanopia.
 *
 * References:
 *  - Okabe & Ito, "Color Universal Design" (https://jfly.uni-koeln.de/color/)
 *  - Paul Tol, "Colour Schemes" (https://personal.sron.nl/~pault/)
 *  - NCEAS colorblind-safe schemes; davidmathlogic.com/colorblind
 */
export const COLORBLIND_COLORS = [
  '#0072B2', // blue
  '#E69F00', // orange
  '#009E73', // bluish green
  '#CC79A7', // reddish purple
  '#56B4E9', // sky blue
  '#D55E00', // vermillion
  '#F0E442', // yellow
  '#999999', // grey
  '#332288', // indigo
  '#117733', // green
  '#AA4499', // purple
  '#88CCEE', // cyan
];

/** The palette to use for new class colors, based on the current preference. */
export function getClassPalette(): string[] {
  return useSettingsStore.getState().colorblindMode ? COLORBLIND_COLORS : DEFAULT_COLORS;
}

/**
 * Pick the first palette color not already in `usedColors`; if all are taken,
 * cycle back using `fallbackIndex`.
 */
export function pickNextColor(usedColors: Iterable<string>, fallbackIndex: number): string {
  const palette = getClassPalette();
  const used = new Set(usedColors);
  return palette.find((c) => !used.has(c)) ?? palette[fallbackIndex % palette.length];
}
