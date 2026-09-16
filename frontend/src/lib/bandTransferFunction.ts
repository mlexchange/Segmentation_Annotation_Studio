/**
 * bandTransferFunction — bridges the Sampler lasso's fitted intensity band
 * (`thresholdFit.ts`'s `BandFit.lo`/`hi`) into the 3D viewer's opacity-curve
 * transfer function, so a sparse, materially-distinct feature can be
 * isolated in 3D by intensity alone — opaque inside the band, transparent
 * outside — with zero upstream viewer changes (see the vendored
 * `OpacityPoint = [intensity: 0-1, opacity: 0-1]` curve format).
 *
 * Deliberately narrow in scope: this only isolates by intensity VALUE across
 * the whole volume, not by the traced SPATIAL region specifically. Good
 * enough for a feature whose density is genuinely distinct from its
 * surroundings (the same property the 2D threshold-lasso tool already
 * exploits) — see the plan's own trade-off note.
 *
 * A real gotcha this module exists to fix: `BandFit.lo`/`hi` are 0-255 BYTE
 * values from the 2D Annotate canvas's own PNG rendering — normalized via
 * `GET /api/image/slice`'s norm="global" percentile stretch (see
 * `ImageMeta.globalValueRange`), NOT the volume's raw physical intensity.
 * The 3D viewer separately normalizes raw voxel values into its own [0,1]
 * domain against a DIFFERENT per-dataset range (`WebGpuViewerInstance
 * .getValueRange()`, a min/max-based estimate — see that method's own doc).
 * These two ranges are typically close but never identical (different
 * statistics, computed from different data), so a fitted band must be
 * converted byte -> raw physical value -> 3D domain, not just divided by
 * 255 — dividing by 255 assumes the two normalizations are the same, which
 * silently produces a wildly wrong band (confirmed live: a materially-sparse
 * 2D-highlighted feature came out as almost the entire volume in 3D).
 */

export type OpacityPoint = readonly [intensity: number, opacity: number];

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/**
 * Converts a fitted band's byte bounds (0-255, the 2D canvas's own
 * percentile-stretched display space) into the 3D viewer's [0,1] domain.
 *
 * `globalRange` is `ImageMeta.globalValueRange` — the `[vmin, vmax]` the 2D
 * backend actually stretched into 0-255; falls back to treating the byte
 * scale as already-physical (`[0, 255]`) if unavailable (e.g. an RGB source,
 * which the backend doesn't compute this for), matching this module's older,
 * pre-conversion behavior.
 *
 * `viewerRange` is `WebGpuViewerInstance.getValueRange()` — the actual range
 * the 3D viewer normalizes raw voxels against for this dataset.
 */
export function mapByteBandToViewerDomain(
  loByte: number,
  hiByte: number,
  globalRange: readonly [number, number] | null | undefined,
  viewerRange: readonly [number, number],
): [number, number] {
  const [gMin, gMax] =
    globalRange && Number.isFinite(globalRange[0]) && Number.isFinite(globalRange[1]) && globalRange[1] > globalRange[0]
      ? globalRange
      : [0, 255];
  const toRaw = (byte: number) => gMin + (byte / 255) * (gMax - gMin);

  const [vMin, vMax] = viewerRange;
  const span = vMax - vMin || 1;
  const toDomain = (raw: number) => clamp01((raw - vMin) / span);

  return [toDomain(toRaw(loByte)), toDomain(toRaw(hiByte))];
}

/**
 * Builds an opacity curve that ramps to fully opaque across `[lo01, hi01]`
 * (already in the viewer's own [0,1] domain — see `mapByteBandToViewerDomain`)
 * and stays transparent everywhere else, with a small ramp at each edge
 * (rather than a hard step) so the transfer function doesn't alias.
 */
export function buildBandOpacityCurve(lo01: number, hi01: number): OpacityPoint[] {
  const EPS = 0.004;
  const loN = clamp01(Math.min(lo01, hi01));
  const hiN = clamp01(Math.max(lo01, hi01));

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
