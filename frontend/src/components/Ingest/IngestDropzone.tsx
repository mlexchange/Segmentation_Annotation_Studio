/**
 * IngestDropzone — drag-and-drop (or pick) image files/folders and copy them
 * into the connected Tiled server as individually-browsable samples.
 *
 * Shown on the Connect page once a Tiled server is selected. On success it can
 * point the Browse view at the freshly-created container.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { UploadSimple, Warning, CheckCircle, CaretRight, CaretDown } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import ConflictDialog, { type IngestConflict } from './ConflictDialog';
import { summarizeIngestErrors, type IngestError } from './ingestErrors';

const SUPPORTED_EXTS = ['tif', 'tiff', 'npy', 'png', 'jpg', 'jpeg'];

interface JobStatus {
  state: 'pending' | 'running' | 'done' | 'error';
  total: number;
  done: number;
  failed: number;
  /** Duplicates left untouched because the user chose "skip". */
  skipped: number;
  errors: IngestError[];
  container_path: string;
  /** Container-relative path of the first successfully-ingested array, e.g.
   *  `almond_control/slice_01` under grouped uploads or plain `slice_01` when
   *  ungrouped — set by the backend, not guessed, so it's correct even when
   *  grouping puts arrays a level deeper than their filename stem suggests. */
  first_path?: string | null;
}

/** How the batch's images map onto samples (see backend/ingest.py sample_name_for). */
type Grouping = 'prefix' | 'per_image' | 'single';

const GROUPING_OPTIONS: { value: Grouping; label: string; hint: string }[] = [
  {
    value: 'prefix',
    label: 'Group by filename, before the “__”',
    hint: 'almond_control__slice_01 → sample “almond_control”. Files without “__” stay together as one sample.',
  },
  { value: 'per_image', label: 'Each image is its own sample', hint: 'One sample per file.' },
  {
    value: 'single',
    label: 'All these images are one sample',
    hint: 'For a z-stack or time series of a single specimen.',
  },
];

/** Response of POST /api/ingest/preflight — what already exists at the destination. */
interface Preflight {
  container_exists: boolean;
  existing_count: number;
  conflicts: IngestConflict[];
  suggested_container_path: string;
  /** How the current grouping would split this batch. */
  sample_preview?: { sample: string; n_images: number }[];
}

/** An upload held back while the user resolves a collision in ConflictDialog. */
interface PendingUpload {
  files: File[];
  target: string;
  preflight: Preflight;
}

/** How the backend should treat node keys that already exist. */
type OnConflict = 'fail' | 'replace' | 'skip';

interface IngestDropzoneProps {
  serverUri: string;
  /** Called when the user wants to browse the freshly-ingested container. */
  onBrowse?: (containerPath: string, sampleCount: number) => void;
  /** Called to open the first ingested sample directly in the Annotate tab. */
  onAnnotate?: (containerPath: string, firstKey: string) => void;
}

function isSupported(name: string): boolean {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  return SUPPORTED_EXTS.includes(ext);
}

/** Tiled node key the backend assigns to a file = its filename stem. */
function fileStem(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

/** Turn a name into a filesystem/Tiled-safe slug, defaulting to 'dataset' if empty. */
function sanitizeName(name: string): string {
  return name.replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset';
}

/** Recursively read a dropped directory entry into a flat list of Files. */
async function readEntry(entry: any): Promise<File[]> {
  if (entry.isFile) {
    return new Promise((resolve) => entry.file((f: File) => resolve([f]), () => resolve([])));
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const all: File[] = [];
    // readEntries returns batches; keep calling until empty.
    const readBatch = (): Promise<void> =>
      new Promise((resolve) => {
        reader.readEntries(async (entries: any[]) => {
          if (!entries.length) return resolve();
          for (const e of entries) all.push(...(await readEntry(e)));
          await readBatch();
          resolve();
        }, () => resolve());
      });
    await readBatch();
    return all;
  }
  return [];
}

/** Gather all dropped files (recursing into directories) and the top-level folder name. */
async function collectFromDrop(dt: DataTransfer): Promise<{ files: File[]; folderName: string }> {
  const items = Array.from(dt.items);
  const entries = items
    .map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null))
    .filter(Boolean) as any[];

  let folderName = '';
  if (entries.length) {
    const dir = entries.find((e) => e.isDirectory);
    if (dir) folderName = dir.name;
    const nested = await Promise.all(entries.map((e) => readEntry(e)));
    return { files: nested.flat(), folderName };
  }
  // Fallback: plain file list (no directory structure).
  return { files: Array.from(dt.files), folderName };
}

export default function IngestDropzone({ serverUri, onBrowse, onAnnotate }: IngestDropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const [containerPath, setContainerPath] = useState('browse/');
  const [description, setDescription] = useState('');
  const [grouping, setGrouping] = useState<Grouping>('prefix');
  // "Save uploaded images to" is optional (auto-derived), so keep it collapsed.
  const [showDest, setShowDest] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  // True while the pre-flight duplicate check is in flight (before any upload).
  const [checking, setChecking] = useState(false);
  // Node key of the first ingested sample (for "Open in Annotate").
  const [firstKey, setFirstKey] = useState<string | null>(null);
  // Set when duplicates are found; renders the ConflictDialog.
  const [pending, setPending] = useState<PendingUpload | null>(null);
  // Files of the in-flight/just-finished batch, kept so the post-upload conflict
  // backstop can re-upload them without the user re-dropping the folder.
  const [lastBatch, setLastBatch] = useState<{ files: File[]; target: string } | null>(null);
  // Indices of error groups whose filenames are expanded.
  const [expandedErrors, setExpandedErrors] = useState<Set<number>>(new Set());
  const pollRef = useRef<number | null>(null);

  /** Clear the active status-polling interval, if any. */
  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  /** Poll the ingest job status every second until it reports done or error. */
  const poll = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = window.setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/ingest/status/${id}`);
          if (!res.ok) throw new Error(await res.text());
          const job: JobStatus = await res.json();
          setStatus(job);
          if (job.state === 'done' || job.state === 'error') stopPolling();
        } catch (e) {
          setError(String(e));
          stopPolling();
        }
      }, 1000);
    },
    [stopPolling],
  );

  /**
   * Ask the server which of *names* already exist in *target*.
   *
   * POST, not GET: a dropped folder can hold hundreds of filenames, and as query
   * params those blow past the HTTP header size limit (431) before the request is
   * ever routed. Pass an empty *names* to only ask for a free container name.
   */
  const requestPreflight = useCallback(
    async (target: string, names: string[]): Promise<Preflight> => {
      const res = await fetch(`${API_BASE}/api/ingest/preflight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ container_path: target, names, server_uri: serverUri, grouping }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    [serverUri, grouping],
  );

  /**
   * POST the files to the ingest endpoint and start polling the resulting job.
   *
   * @param onConflict How the backend should treat keys that already exist —
   *   only ever anything but 'fail' after the user chose in ConflictDialog.
   */
  const doUpload = useCallback(
    async (files: File[], target: string, onConflict: OnConflict) => {
      setPending(null);
      setError(null);
      setStatus(null);
      setExpandedErrors(new Set());
      setLastBatch({ files, target });
      setUploading(true);
      try {
        const fd = new FormData();
        fd.append('container_path', target);
        fd.append('on_conflict', onConflict);
        fd.append('grouping', grouping);
        if (description.trim()) fd.append('description', description.trim());
        for (const f of files) fd.append('files', f, f.name);

        const res = await fetch(
          `${API_BASE}/api/ingest/upload?server_uri=${encodeURIComponent(serverUri)}`,
          { method: 'POST', body: fd },
        );
        if (!res.ok) throw new Error(await res.text());
        const { job_id } = await res.json();
        setJobId(job_id);
        setStatus({
          state: 'running',
          total: files.length,
          done: 0,
          failed: 0,
          skipped: 0,
          errors: [],
          container_path: target,
        });
        poll(job_id);
      } catch (e) {
        setError(`Could not start the upload: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setUploading(false);
      }
    },
    [description, serverUri, poll, grouping],
  );

  /**
   * Filter to supported files, derive a destination container name, then ask the
   * server whether any of these samples already exist there. Clean → upload
   * straight away; duplicates → hand off to ConflictDialog so the user decides.
   */
  const prepareUpload = useCallback(
    async (files: File[], suggestedName: string) => {
      const supported = files.filter((f) => isSupported(f.name));
      if (supported.length === 0) {
        setError('No supported image files (TIFF, PNG, JPG, NPY).');
        return;
      }
      const sortedNames = [...supported].map((f) => f.name).sort();

      // Default the destination name: dropped-folder name → single file's stem →
      // generic 'dataset'. The user can still override the field before/after.
      let target = containerPath.trim();
      if (!target || target === 'browse/') {
        const base =
          suggestedName || (supported.length === 1 ? fileStem(sortedNames[0]) : 'dataset');
        target = `browse/${sanitizeName(base)}`;
        setContainerPath(target);
      }

      // Remember the first sample (by name) so we can jump straight to Annotate.
      setFirstKey(fileStem(sortedNames[0]));
      setError(null);

      setChecking(true);
      try {
        const pre = await requestPreflight(target, supported.map((f) => f.name));
        if (pre.conflicts.length > 0) {
          setPending({ files: supported, target, preflight: pre });
          return;
        }
      } catch (e) {
        // Pre-flight is an optimization; never block the upload on it. If it did
        // fail, the post-upload backstop below still offers the same choices.
        console.warn('ingest pre-flight failed, uploading anyway', e);
      } finally {
        setChecking(false);
      }
      await doUpload(supported, target, 'fail');
    },
    [containerPath, requestPreflight, doUpload],
  );

  /**
   * Backstop for when pre-flight didn't run (or couldn't): if a finished job
   * reports conflicts, offer the same four choices after the fact, re-using the
   * files we still hold so the user never has to re-drop the folder.
   */
  useEffect(() => {
    if (!status || (status.state !== 'done' && status.state !== 'error')) return;
    const conflicts = status.errors.filter((e) => e?.kind === 'conflict');
    if (conflicts.length === 0) {
      setLastBatch(null);
      return;
    }
    if (!lastBatch || lastBatch.target !== status.container_path) return;

    let cancelled = false;
    (async () => {
      // Only for existing_count / a free name suggestion; conflicts are known.
      const pre = await requestPreflight(lastBatch.target, []).catch(() => null);
      if (cancelled) return;
      setPending({
        files: lastBatch.files,
        target: lastBatch.target,
        preflight: {
          container_exists: true,
          existing_count: pre?.existing_count ?? conflicts.length,
          conflicts: conflicts.map((e) => ({ filename: e.filename, key: fileStem(e.filename) })),
          suggested_container_path: pre?.suggested_container_path ?? `${lastBatch.target}_2`,
        },
      });
    })();
    return () => {
      cancelled = true;
    };
    // `pending` is deliberately not a dependency — setting it here must not retrigger.
  }, [status, lastBatch, requestPreflight]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dirInputRef = useRef<HTMLInputElement | null>(null);

  /** Drop handler: collect dropped files/folder and kick off the upload. */
  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      const { files, folderName } = await collectFromDrop(e.dataTransfer);
      await prepareUpload(files, folderName);
    },
    [prepareUpload],
  );

  /** File/folder input change handler: start the upload from the chosen files. */
  const onPick = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      const folderName = files[0]?.webkitRelativePath?.split('/')[0] ?? '';
      prepareUpload(files, folderName);
    },
    [prepareUpload],
  );

  /** Toggle the filename list of one error group. */
  const toggleErrorGroup = useCallback((index: number) => {
    setExpandedErrors((prev) => {
      const next = new Set(prev);
      if (!next.delete(index)) next.add(index);
      return next;
    });
  }, []);

  /** Pre-flight and upload both lock the dropzone. */
  const busy = uploading || checking;
  const errorGroups = status ? summarizeIngestErrors(status.errors) : [];
  const done = status?.state === 'done';
  const failedFatally = status?.state === 'error';
  const processed = status ? status.done + status.failed + status.skipped : 0;
  const pct = status && status.total ? Math.round((processed / status.total) * 100) : 0;

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-sky-100 block">Ingest data into this server</label>

      <label className="block text-xs text-sky-300/80">
        Classes / keywords <span className="text-sky-300/80">(comma-separated, optional)</span>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          disabled={busy}
          placeholder="e.g. air, sample, void, pore"
          className="mt-1 w-full border border-white/20 rounded-md px-2 py-1.5 text-sm bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
        />
        <span className="block mt-1 text-[11px] text-sky-300/80">
          Each entry is pre-created as an annotation class for this dataset in Annotate,
          and becomes an individually-searchable tag in Browse.
        </span>
      </label>

      {/* What counts as a "sample". Browse lists one row per sample, so this decides
          whether a mixed drop becomes one lumped dataset or several — it has to be set
          BEFORE a drop starts the upload, so it sits above the dropzone rather than
          behind the optional destination-path disclosure below (which most people
          never open, by which point the upload has already started). */}
      <fieldset className={uploading || pending ? 'opacity-50' : undefined} disabled={uploading || !!pending}>
        <legend className="text-xs font-medium text-sky-100">These images are…</legend>
        <div className="mt-1 space-y-1.5">
          {GROUPING_OPTIONS.map((opt) => (
            <label key={opt.value} className="flex gap-2 cursor-pointer group">
              <input
                type="radio"
                name="ingest-grouping"
                className="mt-0.5 shrink-0 accent-sky-500"
                checked={grouping === opt.value}
                onChange={() => setGrouping(opt.value)}
              />
              <span className="min-w-0">
                <span className="block text-xs text-white/90">{opt.label}</span>
                <span className="block text-[10px] leading-snug text-sky-300/70">{opt.hint}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <div
        role="button"
        tabIndex={0}
        aria-label="Choose files or a folder to upload"
        onClick={() => !busy && fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !busy) {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!dragging) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setDragging(false);
        }}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 ${
          dragging ? 'border-sky-400 bg-sky-400/10' : 'border-white/20 bg-white/5 hover:border-white/40'
        }`}
      >
        {/* One picker for individual files, one for a whole folder. */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg,.npy"
          onChange={onPick}
          disabled={busy}
          className="hidden"
        />
        <input
          ref={dirInputRef}
          type="file"
          // @ts-expect-error — non-standard but widely supported for folder selection
          webkitdirectory=""
          onChange={onPick}
          disabled={busy}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-2">
          <UploadSimple size={28} className="text-sky-300 pointer-events-none" />
          <p className="text-sm text-sky-100 font-medium pointer-events-none">
            {checking
              ? 'Checking the destination…'
              : uploading
                ? 'Uploading…'
                : 'Drag an image file or folder of images here'}
          </p>
          {!busy && (
            <p className="text-xs text-sky-300/90 flex items-center justify-center gap-1 flex-wrap">
              or
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                className="underline hover:text-sky-200 inline-flex items-center min-h-[24px] px-1"
              >
                choose files
              </button>
              ·
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); dirInputRef.current?.click(); }}
                className="underline hover:text-sky-200 inline-flex items-center min-h-[24px] px-1"
              >
                choose a folder
              </button>
            </p>
          )}
          <p className="text-[10px] text-sky-300/80 pointer-events-none">TIFF, PNG, JPG, NPY</p>
        </div>
      </div>

      <div className="text-xs text-sky-300/80">
        <button
          type="button"
          onClick={() => setShowDest((v) => !v)}
          className="flex items-center gap-1 hover:text-sky-100 transition-colors"
        >
          {showDest ? <CaretDown size={12} /> : <CaretRight size={12} />}
          Save uploaded images to <span className="text-sky-300/80">(optional)</span>
        </button>
        {showDest && (
          <>
            <input
              type="text"
              value={containerPath}
              onChange={(e) => setContainerPath(e.target.value)}
              placeholder="browse/my_dataset"
              className="mt-1 w-full border border-white/20 rounded-md px-2 py-1.5 text-sm font-mono bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
            />
            <span className="block mt-1 text-[11px] text-sky-300/80">
              Destination path on the server (created if new). Auto-filled from the dropped
              file/folder — edit if you want.
            </span>
          </>
        )}
      </div>

      {status && (
        <div className="space-y-1.5">
          <div className="h-2 w-full rounded bg-white/10 overflow-hidden">
            <div
              className={`h-full transition-all ${failedFatally ? 'bg-red-500' : 'bg-sky-500'}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-xs text-sky-200">
            {done ? (
              <span className="flex items-center gap-1 text-green-300">
                <CheckCircle size={13} /> Ingested {status.done} of {status.total}
                {status.skipped > 0 && (
                  <span className="text-sky-300"> · {status.skipped} already present</span>
                )}
                {status.failed > 0 && <span className="text-amber-300"> · {status.failed} failed</span>}
              </span>
            ) : failedFatally ? (
              // Nothing landed — never dress this up with a success checkmark.
              <span className="flex items-center gap-1 text-red-300">
                <Warning size={13} /> Nothing was ingested
                {status.failed > 0 && <span> · {status.failed} of {status.total} failed</span>}
              </span>
            ) : (
              <span>
                {processed} / {status.total} processed…
              </span>
            )}
          </p>
          {/* One row per reason: 690 identical failures read as a single line. */}
          {errorGroups.length > 0 && (
            <ul className="text-[11px] text-red-300/90 space-y-1">
              {errorGroups.map((group, i) => (
                <li key={`${group.kind}-${i}`}>
                  <div className="flex items-start gap-1">
                    <span className="flex-1">{group.summary}</span>
                    {group.filenames.length > 1 && (
                      <button
                        type="button"
                        onClick={() => toggleErrorGroup(i)}
                        className="flex shrink-0 items-center gap-0.5 text-red-300/70 hover:text-red-200 transition-colors"
                      >
                        {expandedErrors.has(i) ? <CaretDown size={10} /> : <CaretRight size={10} />}
                        {expandedErrors.has(i) ? 'hide' : 'show'} filenames
                      </button>
                    )}
                  </div>
                  {expandedErrors.has(i) && (
                    <ul className="mt-0.5 max-h-24 overflow-y-auto pl-2 font-mono text-red-300/70">
                      {group.filenames.map((name) => (
                        <li key={name}>{name}</li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          )}
          {/* Skipped duplicates are still in the dataset, so browsing makes sense. */}
          {done && status.done + status.skipped > 0 && (onBrowse || onAnnotate) && (
            <div className="mt-1 flex gap-2">
              {onBrowse && (
                <button
                  type="button"
                  onClick={() => onBrowse(status.container_path, status.done + status.skipped)}
                  className="flex-1 bg-sky-600 text-white rounded-md py-2 text-sm font-medium hover:bg-sky-700 transition-colors"
                >
                  Browse this dataset
                </button>
              )}
              {onAnnotate && (status.first_path || firstKey) && (
                <button
                  type="button"
                  // Prefer the server-reported path: under grouping, the array lives at
                  // `<sample>/<stem>`, one level deeper than the client-guessed stem
                  // alone — `firstKey` is only a fallback for the rare case where
                  // nothing new was written (a wholly-duplicate re-upload), so the
                  // server never got to report a first_path at all.
                  onClick={() => onAnnotate(status.container_path, status.first_path || firstKey!)}
                  className="flex-1 bg-emerald-600 text-white rounded-md py-2 text-sm font-medium hover:bg-emerald-700 transition-colors"
                >
                  {status.done > 1 ? 'Annotate first image' : 'Open in Annotate'}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="flex items-center gap-1.5 text-xs text-red-400">
          <Warning size={13} /> {error}
        </p>
      )}

      {pending && (
        <ConflictDialog
          containerPath={pending.target}
          totalFiles={pending.files.length}
          conflicts={pending.preflight.conflicts}
          existingCount={pending.preflight.existing_count}
          suggestedContainerPath={pending.preflight.suggested_container_path}
          samplePreview={pending.preflight.sample_preview}
          onReplace={() => doUpload(pending.files, pending.target, 'replace')}
          onSkip={() => doUpload(pending.files, pending.target, 'skip')}
          onNewDataset={() => {
            const next = pending.preflight.suggested_container_path;
            setContainerPath(next);
            doUpload(pending.files, next, 'fail');
          }}
          onBrowseExisting={() => {
            setPending(null);
            onBrowse?.(pending.target, pending.preflight.existing_count);
          }}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
}
