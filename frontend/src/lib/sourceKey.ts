/**
 * Canonical source-key helpers.
 *
 * A "source key" uniquely identifies an annotated sample for draft storage and
 * for keying the annotation store.  All code must use these helpers so that
 * autosave keys and load keys always match.
 *
 *   tiled  → "tiled:<serverUri>:<tiledPath>"
 *   local  → "local:<relPath>"
 */

/** Builds the canonical sourceKey for a sample; tiled keys embed the server URI, local keys use the relative path. */
export function buildSourceKey(
  kind: 'tiled' | 'local',
  path: string,
  serverUri?: string | null,
): string {
  return kind === 'tiled' ? `tiled:${serverUri ?? ''}:${path}` : `local:${path}`;
}

export interface ParsedSourceKey {
  kind: 'tiled' | 'local';
  path: string;
  serverUri: string | null;
}

// Legacy keys are intentionally retained because they are persisted by the backend.
// Parse the URL authority as a unit instead of splitting at the first `:`, which
// corrupts every http(s) URI and is especially unsafe for ports and IPv6 literals.
const LEGACY_TILED_KEY = /^(https?:\/\/(?:\[[^\]]+\]|[^/:]+)(?::\d+)?):(.*)$/u;

/** Parse a persisted source key without losing URI ports, IPv6 literals, or path colons. */
export function parseSourceKey(sourceKey: string): ParsedSourceKey {
  if (sourceKey.startsWith('local:')) {
    return { kind: 'local', path: sourceKey.slice('local:'.length), serverUri: null };
  }

  if (!sourceKey.startsWith('tiled:')) {
    throw new Error(`Invalid source key: ${sourceKey}`);
  }

  const rest = sourceKey.slice('tiled:'.length);
  if (rest.startsWith(':')) {
    return { kind: 'tiled', path: rest.slice(1), serverUri: null };
  }

  const match = LEGACY_TILED_KEY.exec(rest);
  if (!match) {
    throw new Error(`Invalid tiled source key: ${sourceKey}`);
  }
  return { kind: 'tiled', serverUri: match[1], path: match[2] };
}

/**
 * True if *path* itself, or anything below it, has annotations.
 *
 * One image can legitimately be keyed two ways: as a whole volume
 * (`browse/ds`, annotated slice-by-slice — what Browse opens from the browse
 * root) or as a standalone array (`browse/ds/img_0001` — what "Annotate first
 * image" and slice-level entry points use). An exact-match lookup therefore
 * leaves a dataset looking untouched when it was annotated through the other
 * entry point, so a container also counts as annotated when a descendant is.
 *
 * Deliberately NOT the reverse: a single slice must not be badged just because
 * some other slice of its volume was annotated — that would light up every
 * frame of a 690-image stack on the strength of one.
 *
 * @param annotatedKeys Source keys known to have annotations.
 * @param path Tiled path of the row being rendered.
 * @param serverUri Server the row belongs to (part of the key).
 */
export function isAnnotatedPath(
  annotatedKeys: Set<string>,
  path: string,
  serverUri?: string | null,
): boolean {
  const self = buildSourceKey('tiled', path, serverUri);
  if (annotatedKeys.has(self)) return true;

  const childPrefix = `${self}/`;
  for (const key of annotatedKeys) {
    if (key.startsWith(childPrefix)) return true;
  }
  return false;
}
