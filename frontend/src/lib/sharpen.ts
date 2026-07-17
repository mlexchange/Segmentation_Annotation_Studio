/**
 * Sharpen — classic single-pass 3×3 Laplacian high-boost (display-only).
 *
 *   [  0 -1  0 ]
 *   [ -1  5 -1 ]
 *   [  0 -1  0 ]
 *
 * Fast and predictable (not an unsharp mask). Edges get crisper for annotation.
 * Mutates the RGBA Uint8ClampedArray in place (alpha untouched); edges replicate
 * (out-of-bounds neighbours use the centre pixel). Pure/DOM-free for testing.
 */
export function applySharpen(data: Uint8ClampedArray, width: number, height: number): void {
  if (width === 0 || height === 0) return;
  const src = new Uint8ClampedArray(data); // read from a copy; write into `data`
  const at = (x: number, y: number, c: number) => src[(y * width + x) * 4 + c];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        const center = at(x, y, c);
        const up = y > 0 ? at(x, y - 1, c) : center;
        const down = y < height - 1 ? at(x, y + 1, c) : center;
        const left = x > 0 ? at(x - 1, y, c) : center;
        const right = x < width - 1 ? at(x + 1, y, c) : center;
        data[o + c] = 5 * center - up - down - left - right; // Uint8ClampedArray clamps
      }
    }
  }
}
