/**
 * Binary-mask morphology for user-facing mask cleanup (fill holes, remove small
 * islands, smooth, grow/shrink). Operates on `Uint8Array` 0/1 masks of size
 * `gw × gh`; pairs with `rasterizeShapes` (shapes → mask) and `maskToPolygons`
 * (mask → shapes). Out-of-bounds neighbors count as background.
 */

/** Fill interior holes: any background pixel not connected to the border → 1. */
export function fillHoles(mask: Uint8Array, gw: number, gh: number): Uint8Array {
  const outside = new Uint8Array(gw * gh);
  const stack: number[] = [];
  const pushBg = (i: number) => { if (!mask[i] && !outside[i]) { outside[i] = 1; stack.push(i); } };
  for (let x = 0; x < gw; x++) { pushBg(x); pushBg((gh - 1) * gw + x); }
  for (let y = 0; y < gh; y++) { pushBg(y * gw); pushBg(y * gw + gw - 1); }
  while (stack.length) {
    const idx = stack.pop()!;
    const x = idx % gw, y = (idx / gw) | 0;
    if (x > 0) pushBg(idx - 1);
    if (x < gw - 1) pushBg(idx + 1);
    if (y > 0) pushBg(idx - gw);
    if (y < gh - 1) pushBg(idx + gw);
  }
  const out = new Uint8Array(gw * gh);
  for (let i = 0; i < out.length; i++) out[i] = mask[i] || !outside[i] ? 1 : 0;
  return out;
}

/** Zero out 4-connected foreground components smaller than *minPx* pixels. */
export function removeSmallComponents(mask: Uint8Array, gw: number, gh: number, minPx: number): Uint8Array {
  const out = new Uint8Array(mask); // copy
  const labels = new Int32Array(gw * gh);
  let next = 1;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p] || labels[p]) continue;
    const lbl = next++;
    const comp: number[] = [];
    const stack = [p];
    labels[p] = lbl;
    while (stack.length) {
      const idx = stack.pop()!;
      comp.push(idx);
      const x = idx % gw, y = (idx / gw) | 0;
      const push = (j: number) => { if (mask[j] && !labels[j]) { labels[j] = lbl; stack.push(j); } };
      if (x > 0) push(idx - 1);
      if (x < gw - 1) push(idx + 1);
      if (y > 0) push(idx - gw);
      if (y < gh - 1) push(idx + gw);
    }
    if (comp.length < minPx) for (const idx of comp) out[idx] = 0;
  }
  return out;
}

/**
 * Keep only the 4-connected component(s) that contain any of `pts` (grid coords);
 * drop everything else. If no point lands on a set pixel, returns the LARGEST
 * component instead (so a box prompt with no seed still yields one clean region).
 * Used by the SAM magic wand "connected regions only" option to strip speckle.
 */
export function keepComponentsAtPoints(
  mask: Uint8Array, gw: number, gh: number, pts: Array<{ x: number; y: number }>,
): Uint8Array {
  const labels = new Int32Array(gw * gh);
  const sizes: number[] = [0];
  let next = 1;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p] || labels[p]) continue;
    const lbl = next++;
    let size = 0;
    const stack = [p];
    labels[p] = lbl;
    while (stack.length) {
      const idx = stack.pop()!;
      size++;
      const x = idx % gw, y = (idx / gw) | 0;
      const push = (j: number) => { if (mask[j] && !labels[j]) { labels[j] = lbl; stack.push(j); } };
      if (x > 0) push(idx - 1);
      if (x < gw - 1) push(idx + 1);
      if (y > 0) push(idx - gw);
      if (y < gh - 1) push(idx + gw);
    }
    sizes[lbl] = size;
  }
  if (next === 1) return new Uint8Array(mask); // empty

  const keep = new Set<number>();
  for (const pt of pts) {
    const gx = Math.max(0, Math.min(gw - 1, Math.round(pt.x)));
    const gy = Math.max(0, Math.min(gh - 1, Math.round(pt.y)));
    const lbl = labels[gy * gw + gx];
    if (lbl) keep.add(lbl);
  }
  if (keep.size === 0) {
    // No prompt hit foreground → keep the largest component.
    let best = 1;
    for (let l = 2; l < next; l++) if (sizes[l] > sizes[best]) best = l;
    keep.add(best);
  }
  const out = new Uint8Array(mask.length);
  for (let p = 0; p < mask.length; p++) if (keep.has(labels[p])) out[p] = 1;
  return out;
}

/** One 8-connected dilation pass (a pixel is set if any neighbor is set). */
function dilateOnce(mask: Uint8Array, gw: number, gh: number): Uint8Array {
  const out = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && !on; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= gh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= gw) continue;
          if (mask[yy * gw + xx]) { on = 1; break; }
        }
      }
      out[y * gw + x] = on;
    }
  }
  return out;
}

/** One 8-connected erosion pass (a pixel stays set only if all neighbors are set). */
function erodeOnce(mask: Uint8Array, gw: number, gh: number): Uint8Array {
  const out = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let all = 1;
      for (let dy = -1; dy <= 1 && all; dy++) {
        const yy = y + dy;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= gw || yy < 0 || yy >= gh || !mask[yy * gw + xx]) { all = 0; break; }
        }
      }
      out[y * gw + x] = all;
    }
  }
  return out;
}

/** Grow the mask by *iters* pixels (8-connected dilation). */
export function dilate(mask: Uint8Array, gw: number, gh: number, iters = 1): Uint8Array {
  let m = mask;
  for (let i = 0; i < iters; i++) m = dilateOnce(m, gw, gh);
  return m;
}

/** Shrink the mask by *iters* pixels (8-connected erosion). */
export function erode(mask: Uint8Array, gw: number, gh: number, iters = 1): Uint8Array {
  let m = mask;
  for (let i = 0; i < iters; i++) m = erodeOnce(m, gw, gh);
  return m;
}

/**
 * Smooth the boundary via a majority filter (a pixel is set when ≥5 of its 3×3
 * neighborhood are set), repeated *iters* times. Removes single-pixel specks
 * and jagged edges without changing overall size much.
 */
export function smooth(mask: Uint8Array, gw: number, gh: number, iters = 1): Uint8Array {
  let m = mask;
  for (let it = 0; it < iters; it++) {
    const out = new Uint8Array(gw * gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        let c = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= gh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= gw) continue;
            c += m[yy * gw + xx];
          }
        }
        out[y * gw + x] = c >= 5 ? 1 : 0;
      }
    }
    m = out;
  }
  return m;
}
