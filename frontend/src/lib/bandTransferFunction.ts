/**
 * bandTransferFunction — bridges the Sampler lasso's fitted intensity band
 * (`thresholdFit.ts`'s `BandFit.lo`/`hi`, in the volume's own 0–255 space)
 * into the 3D viewer's opacity-curve transfer function, so a sparse,
 * materially-distinct feature can be isolated in 3D by intensity alone —
 * opaque inside the band, transparent outside — with zero upstream viewer
 * changes (see the vendored `OpacityPoint = [intensity: 0-1, opacity: 0-1]`
 * curve format).
 *
 * Deliberately narrow in scope: this only isolates by intensity VALUE across
 * the whole volume, not by the traced SPATIAL region specifically. Good
 * enough for a feature whose density is genuinely distinct from its
 * surroundings (the same property the 2D threshold-lasso tool already
 * exploits) — see the plan's own trade-off note.
 */

export type OpacityPoint = readonly [intensity: number, opacity: number];

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/**
 * Builds an opacity curve that ramps to fully opaque across `[lo, hi]`
 * (given in 0–255 space) and stays transparent everywhere else, with a small
 * ramp at each edge (rather than a hard step) so the transfer function
 * doesn't alias.
 */
export function buildBandOpacityCurve(lo255: number, hi255: number): OpacityPoint[] {
  const EPS = 0.004; // ~1 intensity level at 8-bit resolution
  const loN = clamp01(Math.min(lo255, hi255) / 255);
  const hiN = clamp01(Math.max(lo255, hi255) / 255);

  const raw: OpacityPoint[] = [
    [0, 0],
    [loN, loN <= EPS ? 1 : 0],
    [Math.min(1, loN + EPS), 1],
    [Math.max(0, hiN - EPS), 1],
    [hiN, hiN >= 1 - EPS ? 1 : 0],
    [1, hiN >= 1 - EPS ? 1 : 0],
  ];

  // Points must have strictly increasing x for the viewer's curve to be
  // well-defined — clamping near 0/1 can collapse several of the above onto
  // the same x. Sort, then keep only the first point at each x.
  const sorted = [...raw].sort((a, b) => a[0] - b[0]);
  const points: OpacityPoint[] = [];
  for (const p of sorted) {
    if (points.length > 0 && p[0] <= points[points.length - 1][0]) continue;
    points.push(p);
  }
  return points;
}
