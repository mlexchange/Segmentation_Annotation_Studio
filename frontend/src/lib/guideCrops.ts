/**
 * Guide example-crop sanitization.
 *
 * `GuideClass.exampleCrops` is rendered directly as `<img src>` in the Reference
 * and Class Manager pages. A guide can arrive from three places: locally picked
 * files (already safe `data:` URLs from `FileReader`), the backend-generated
 * guide (`/api/guide/generate`), and an imported JSON file a user downloaded from
 * anywhere and chose to load ("Load a shared guide file"). That last path is
 * untrusted content — without validation, a crafted guide could point `<img src>`
 * at an arbitrary external or intranet URL, firing a browser request (leaking the
 * viewer's IP/user agent, or probing internal services) the moment the guide
 * renders, with no user awareness. Only same-document, size-bounded image data
 * URLs are accepted; everything else is silently dropped.
 */

const DATA_URL_IMAGE_RE = /^data:image\/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/]+=*$/;
// ~1.5MB decoded — generous for a class thumbnail, small enough to bound memory.
const MAX_CROP_DATA_URL_LENGTH = 2_000_000;
const MAX_CROPS_PER_CLASS = 50;

/** Keep only well-formed, size-bounded `data:image/...;base64,` crop entries. */
export function sanitizeExampleCrops(crops: unknown): string[] {
  if (!Array.isArray(crops)) return [];
  const safe: string[] = [];
  for (const crop of crops) {
    if (typeof crop !== 'string') continue;
    if (crop.length === 0 || crop.length > MAX_CROP_DATA_URL_LENGTH) continue;
    if (!DATA_URL_IMAGE_RE.test(crop)) continue;
    safe.push(crop);
    if (safe.length >= MAX_CROPS_PER_CLASS) break;
  }
  return safe;
}
