/**
 * Human-readable copy for the classified per-file errors an ingest job reports.
 *
 * The backend (see ``_classify_error`` in backend/ingest.py) sends a stable
 * `kind` token plus a short fallback message, deliberately stripped of internal
 * API URLs. This module owns the user-facing wording.
 */

export type IngestErrorKind =
  | 'conflict'
  | 'unreadable'
  | 'unreachable'
  | 'auth'
  | 'unknown';

export interface IngestError {
  /** Original filename, or '' for a job-level failure. */
  filename: string;
  kind: IngestErrorKind;
  /** Backend fallback text; used when the kind carries no specific copy. */
  message: string;
}

/** One line of the error list: a reason plus every file that hit it. */
export interface IngestErrorGroup {
  kind: IngestErrorKind;
  /** Sentence describing the reason, already pluralized for `filenames`. */
  summary: string;
  filenames: string[];
}

/** Per-file copy. `unknown` has none — the backend's own message is better. */
const COPY: Partial<Record<IngestErrorKind, string>> = {
  conflict: 'already exists in this dataset',
  unreadable: 'could not be read as an image',
  unreachable: 'could not reach the Tiled server',
  auth: 'not authorized to write to this server',
};

/** Copy for a whole group, as `<n> image(s) <this>`. */
const GROUP_COPY: Partial<Record<IngestErrorKind, string>> = {
  conflict: 'already exist in this dataset',
  unreadable: 'could not be read as images',
  unreachable: 'could not be sent — the Tiled server was unreachable',
  auth: 'were rejected — not authorized to write to this server',
};

const GENERIC = 'could not be ingested';

/**
 * Normalize an error entry. Older backends sent plain strings, and an entry from
 * a newer one may carry a kind we don't know — render those verbatim rather than
 * flattening every row to a useless generic sentence.
 */
function normalize(err: IngestError | string): IngestError {
  if (typeof err === 'string') return { filename: '', kind: 'unknown', message: err };
  return {
    filename: err?.filename ?? '',
    kind: err?.kind ?? 'unknown',
    message: err?.message ?? '',
  };
}

/** Return a one-line sentence for an ingest error, prefixed by its filename. */
export function describeIngestError(err: IngestError | string): string {
  const { filename, kind, message } = normalize(err);
  const detail = COPY[kind] ?? message ?? '';
  return filename ? `${filename} — ${detail || GENERIC}` : detail || message || GENERIC;
}

/**
 * Collapse an error list into one row per reason, so a 690-file batch that all
 * failed the same way reads as a single sentence instead of five identical rows.
 *
 * Errors of an unrecognized kind keep their own distinct message, so genuinely
 * different failures never get merged.
 *
 * @returns Groups in first-seen order, each with the filenames that hit it.
 */
export function summarizeIngestErrors(errors: (IngestError | string)[]): IngestErrorGroup[] {
  const groups = new Map<string, IngestError[]>();
  for (const raw of errors) {
    const err = normalize(raw);
    // Known kinds merge by kind; unknown ones merge only by identical message.
    const key = GROUP_COPY[err.kind] ? err.kind : `${err.kind}:${err.message}`;
    const bucket = groups.get(key);
    if (bucket) bucket.push(err);
    else groups.set(key, [err]);
  }

  return [...groups.values()].map((entries) => {
    const { kind, message } = entries[0];
    const reason = GROUP_COPY[kind] ?? (message ? `${GENERIC} — ${message}` : GENERIC);
    return {
      kind,
      // A lone failure reads better as itself than as "1 image …".
      summary:
        entries.length === 1 ? describeIngestError(entries[0]) : `${entries.length} images ${reason}`,
      filenames: entries.map((e) => e.filename).filter(Boolean),
    };
  });
}
