/**
 * Gaussian blur (display-only) — three successive box blurs, the standard
 * approximation of a true Gaussian (by the central limit theorem, three passes
 * are visually indistinguishable from a real Gaussian for σ ≳ 1 while staying
 * O(n) per pass regardless of radius).
 *
 * Used as the first nonlinear display preprocessor so the intensity-driven tools
 * (threshold brush, magic wand, fill) see a denoised image and similar features
 * cohere into one selectable region. Mutates the RGBA Uint8ClampedArray in place
 * (alpha untouched); edges clamp (out-of-bounds samples replicate the edge
 * pixel). Pure/DOM-free for testing.
 */

/** Box radii whose 3-pass cascade approximates a Gaussian of the given sigma.
 *  Standard Kovesi/Gwosdek construction: pick w_ideal from the box variance
 *  identity, split into `m` boxes of size wl and `3-m` of size wl+2. */
function boxRadiiForGaussian(sigma: number, passes: number): number[] {
  const wIdeal = Math.sqrt((12 * sigma * sigma) / passes + 1);
  let wl = Math.floor(wIdeal);
  if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const mIdeal =
    (12 * sigma * sigma - passes * wl * wl - 4 * passes * wl - 3 * passes) /
    (-4 * wl - 4);
  const m = Math.round(mIdeal);
  const radii: number[] = [];
  for (let i = 0; i < passes; i++) {
    const w = i < m ? wl : wu;
    radii.push(Math.max(0, (w - 1) / 2));
  }
  return radii;
}

/** One horizontal box blur of radius `r` from `src` into `dst` (RGB only). */
function boxBlurH(
  src: Uint8ClampedArray,
  dst: Uint8ClampedArray,
  width: number,
  height: number,
  r: number,
): void {
  const norm = 1 / (r + r + 1);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let c = 0; c < 3; c++) {
      // Seed the running sum for x=0 with edge-clamped left neighbours.
      const first = src[row * 4 + c];
      let sum = (r + 1) * first;
      for (let i = 0; i < r; i++) sum += src[(row + Math.min(i, width - 1)) * 4 + c];
      for (let x = 0; x < width; x++) {
        const inIdx = Math.min(x + r, width - 1);
        const outIdx = x - r - 1;
        sum += src[(row + inIdx) * 4 + c];
        sum -= outIdx < 0 ? first : src[(row + outIdx) * 4 + c];
        dst[(row + x) * 4 + c] = sum * norm;
      }
    }
  }
}

/** One vertical box blur of radius `r` from `src` into `dst` (RGB only). */
function boxBlurV(
  src: Uint8ClampedArray,
  dst: Uint8ClampedArray,
  width: number,
  height: number,
  r: number,
): void {
  const norm = 1 / (r + r + 1);
  for (let x = 0; x < width; x++) {
    for (let c = 0; c < 3; c++) {
      const first = src[x * 4 + c];
      let sum = (r + 1) * first;
      for (let i = 0; i < r; i++) sum += src[(Math.min(i, height - 1) * width + x) * 4 + c];
      for (let y = 0; y < height; y++) {
        const inIdx = Math.min(y + r, height - 1);
        const outIdx = y - r - 1;
        sum += src[(inIdx * width + x) * 4 + c];
        sum -= outIdx < 0 ? first : src[(outIdx * width + x) * 4 + c];
        dst[(y * width + x) * 4 + c] = sum * norm;
      }
    }
  }
}

/**
 * Blur a single-channel float field in place, same Gaussian approximation as the
 * RGBA version above.
 *
 * The Threshold Brush's field is already grayscale, so the sampler's blur sweep
 * can work on it directly instead of round-tripping through RGBA — which also
 * means a candidate sigma is evaluated on exactly the values the brush gates on.
 */
export function applyGaussianBlurGray(
  data: Float32Array,
  width: number,
  height: number,
  sigma: number,
): void {
  if (!(sigma > 0) || width === 0 || height === 0) return;
  const scratch = new Float32Array(data.length);

  const boxH = (src: Float32Array, dst: Float32Array, r: number) => {
    const norm = 1 / (r + r + 1);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      const first = src[row];
      let sum = (r + 1) * first;
      for (let i = 0; i < r; i++) sum += src[row + Math.min(i, width - 1)];
      for (let x = 0; x < width; x++) {
        sum += src[row + Math.min(x + r, width - 1)];
        sum -= x - r - 1 < 0 ? first : src[row + x - r - 1];
        dst[row + x] = sum * norm;
      }
    }
  };
  const boxV = (src: Float32Array, dst: Float32Array, r: number) => {
    const norm = 1 / (r + r + 1);
    for (let x = 0; x < width; x++) {
      const first = src[x];
      let sum = (r + 1) * first;
      for (let i = 0; i < r; i++) sum += src[Math.min(i, height - 1) * width + x];
      for (let y = 0; y < height; y++) {
        sum += src[Math.min(y + r, height - 1) * width + x];
        sum -= y - r - 1 < 0 ? first : src[(y - r - 1) * width + x];
        dst[y * width + x] = sum * norm;
      }
    }
  };

  for (const r of boxRadiiForGaussian(sigma, 3)) {
    if (r <= 0) continue;
    boxH(data, scratch, r);
    boxV(scratch, data, r);
  }
}

/**
 * Blur `data` (RGBA, row-major `width × height`) in place by a Gaussian of the
 * given sigma in pixels. No-op for sigma ≤ 0 or an empty image.
 */
export function applyGaussianBlurRgba(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  sigma: number,
): void {
  if (!(sigma > 0) || width === 0 || height === 0) return;
  const radii = boxRadiiForGaussian(sigma, 3);
  // Scratch buffer for the horizontal half of each pass; the vertical half
  // writes straight back into `data`, so the result always lands in place.
  const scratch = new Uint8ClampedArray(data);
  for (const r of radii) {
    if (r <= 0) continue;
    boxBlurH(data, scratch, width, height, r);
    boxBlurV(scratch, data, width, height, r);
  }
}
