/**
 * apiError — turn a FastAPI error body into something a user can act on.
 *
 * FastAPI reports a rejected request as `{"detail": [{loc, msg, type, ...}]}`.
 * Rendered raw that reads as a wall of JSON, which is what the job progress bar
 * used to show. This extracts the field name and message instead.
 */

interface ValidationItem {
  loc?: unknown[];
  msg?: string;
}

const TRUNCATE_AT = 500;

/** Cut *text* to at most `limit` characters, marking that it was cut — a bare
 *  slice reads as the whole message, hiding that the rest was thrown away. */
function truncate(text: string, limit = TRUNCATE_AT): string {
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
}

/**
 * Human-readable name for a pydantic `loc`, e.g. `image_size`, or `sources[2]`
 * when the location is a whole list element rather than one of its fields
 * (there's no field name to fall back on there, just the index).
 */
function fieldName(loc: unknown[] | undefined): string | null {
  if (!Array.isArray(loc)) return null;
  // Drop the leading "body" and any discriminated-union tag; a bare numeric
  // index (a list position) is kept only long enough to attach it to the
  // string segment right before it, then treated as the leaf itself.
  const parts = loc.filter(
    (p): p is string | number => p !== 'body' && (typeof p === 'string' || typeof p === 'number'),
  );
  const leaf = parts[parts.length - 1];
  if (leaf === undefined) return null;
  if (typeof leaf === 'number') {
    const prev = parts[parts.length - 2];
    return (typeof prev === 'string' ? `${prev}[${leaf}]` : `item ${leaf}`).replace(/_/g, ' ');
  }
  return leaf.replace(/_/g, ' ');
}

/** One `{msg, loc}`-shaped validation item (or a plain string) to a display line. */
function formatValidationItem(item: unknown): string | null {
  if (typeof item === 'string') return item.trim() || null;
  const msg = typeof (item as ValidationItem)?.msg === 'string' ? (item as ValidationItem).msg! : '';
  if (!msg) return null;
  const field = fieldName((item as ValidationItem)?.loc);
  return field ? `${field}: ${msg}` : msg;
}

/**
 * Best-effort human-readable message for a non-OK API response body.
 *
 * Falls back to the raw text (trimmed) when the body isn't a shape we recognise,
 * so nothing is ever swallowed — an unexpected error still reaches the user.
 */
export function formatApiError(body: string, fallback = 'Request failed.'): string {
  const text = (body ?? '').trim();
  if (!text) return fallback;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return truncate(text);
  }

  const detail = (parsed as { detail?: unknown })?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;

  // The common shape: a list of pydantic validation errors (or, less commonly,
  // a list of plain message strings — some routes raise HTTPException with one).
  if (Array.isArray(detail)) {
    const messages = detail.map(formatValidationItem).filter((m): m is string => Boolean(m));
    if (messages.length) return messages.join('; ');
  }

  // A single validation error/object, not wrapped in a list — same shape either way.
  if (detail && typeof detail === 'object') {
    const message = formatValidationItem(detail);
    if (message) return message;
  }

  return truncate(text);
}
