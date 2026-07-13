/**
 * Colormaps / LUTs for scientific display. Each map is a small set of RGB
 * control points (0–1); SVG `feFunc type="table"` interpolates between them,
 * so we can false-color a grayscale slice on the GPU with no per-pixel JS.
 *
 * Display-only: colormaps affect what the eye sees, never exported pixels or the
 * intensity field the wand/SAM tools operate on.
 */

export type ColormapName = 'gray' | 'viridis' | 'magma' | 'inferno';

export const COLORMAP_NAMES: ColormapName[] = ['gray', 'viridis', 'magma', 'inferno'];

/** RGB control points (0–1), low→high intensity. */
const STOPS: Record<Exclude<ColormapName, 'gray'>, [number, number, number][]> = {
  viridis: [
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
  ],
  magma: [
    [0.001, 0.000, 0.014], [0.109, 0.065, 0.276], [0.316, 0.072, 0.485],
    [0.508, 0.146, 0.505], [0.710, 0.212, 0.478], [0.898, 0.315, 0.396],
    [0.984, 0.529, 0.381], [0.996, 0.760, 0.529], [0.987, 0.991, 0.750],
  ],
  inferno: [
    [0.001, 0.000, 0.014], [0.146, 0.046, 0.359], [0.339, 0.062, 0.429],
    [0.531, 0.132, 0.416], [0.720, 0.215, 0.330], [0.878, 0.336, 0.198],
    [0.972, 0.522, 0.086], [0.988, 0.749, 0.185], [0.988, 0.998, 0.645],
  ],
};

/** Per-channel tableValues for `feFunc type="table"`, or null for identity (gray). */
export function colormapTables(
  name: ColormapName,
): { r: number[]; g: number[]; b: number[] } | null {
  if (name === 'gray') return null;
  const stops = STOPS[name];
  return {
    r: stops.map((s) => s[0]),
    g: stops.map((s) => s[1]),
    b: stops.map((s) => s[2]),
  };
}

/** CSS linear-gradient string for a colormap swatch (low→high, left→right). */
export function colormapGradient(name: ColormapName): string {
  if (name === 'gray') return 'linear-gradient(to right, #000, #fff)';
  const stops = STOPS[name];
  const parts = stops.map((s, i) => {
    const pct = Math.round((i / (stops.length - 1)) * 100);
    const rgb = `rgb(${Math.round(s[0] * 255)},${Math.round(s[1] * 255)},${Math.round(s[2] * 255)})`;
    return `${rgb} ${pct}%`;
  });
  return `linear-gradient(to right, ${parts.join(', ')})`;
}
