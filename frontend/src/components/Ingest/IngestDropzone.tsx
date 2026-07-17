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

const SUPPORTED_EXTS = ['tif', 'tiff', 'npy', 'png', 'jpg', 'jpeg'];

interface JobStatus {
  state: 'pending' | 'running' | 'done' | 'error';
  total: number;
  done: number;
  failed: number;
  errors: string[];
  container_path: string;
}

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
  // "Save uploaded images to" is optional (auto-derived), so keep it collapsed.
  const [showDest, setShowDest] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  // Node key of the first ingested sample (for "Open in Annotate").
  const [firstKey, setFirstKey] = useState<string | null>(null);
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
   * Filter to supported files, derive a destination container name, POST the upload,
   * then start polling the resulting job. Side-effects: sets status/error/firstKey state.
   */
  const startUpload = useCallback(
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
      setStatus(null);
      setUploading(true);
      try {
        const fd = new FormData();
        fd.append('container_path', target);
        if (description.trim()) fd.append('description', description.trim());
        for (const f of supported) fd.append('files', f, f.name);

        const res = await fetch(
          `${API_BASE}/api/ingest/upload?server_uri=${encodeURIComponent(serverUri)}`,
          { method: 'POST', body: fd },
        );
        if (!res.ok) throw new Error(await res.text());
        const { job_id } = await res.json();
        setJobId(job_id);
        setStatus({ state: 'running', total: supported.length, done: 0, failed: 0, errors: [], container_path: target });
        poll(job_id);
      } catch (e) {
        setError(String(e));
      } finally {
        setUploading(false);
      }
    },
    [containerPath, description, serverUri, poll],
  );

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dirInputRef = useRef<HTMLInputElement | null>(null);

  /** Drop handler: collect dropped files/folder and kick off the upload. */
  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      const { files, folderName } = await collectFromDrop(e.dataTransfer);
      await startUpload(files, folderName);
    },
    [startUpload],
  );

  /** File/folder input change handler: start the upload from the chosen files. */
  const onPick = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      const folderName = files[0]?.webkitRelativePath?.split('/')[0] ?? '';
      startUpload(files, folderName);
    },
    [startUpload],
  );

  const done = status?.state === 'done';
  const failedFatally = status?.state === 'error';
  const processed = status ? status.done + status.failed : 0;
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
          disabled={uploading}
          placeholder="e.g. air, sample, void, pore"
          className="mt-1 w-full border border-white/20 rounded-md px-2 py-1.5 text-sm bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
        />
        <span className="block mt-1 text-[11px] text-sky-300/80">
          Each entry is pre-created as an annotation class for this dataset in Annotate,
          and becomes an individually-searchable tag in Browse.
        </span>
      </label>

      <div
        role="button"
        tabIndex={0}
        onClick={() => !uploading && fileInputRef.current?.click()}
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
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${
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
          disabled={uploading}
          className="hidden"
        />
        <input
          ref={dirInputRef}
          type="file"
          // @ts-expect-error — non-standard but widely supported for folder selection
          webkitdirectory=""
          onChange={onPick}
          disabled={uploading}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-2">
          <UploadSimple size={28} className="text-sky-300 pointer-events-none" />
          <p className="text-sm text-sky-100 font-medium pointer-events-none">
            {uploading ? 'Uploading…' : 'Drag an image file or folder of images here'}
          </p>
          {!uploading && (
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
                {status.failed > 0 && <span className="text-amber-300"> · {status.failed} failed</span>}
              </span>
            ) : failedFatally ? (
              <span className="text-red-300">Ingest failed</span>
            ) : (
              <span>
                {processed} / {status.total} processed…
              </span>
            )}
          </p>
          {status.errors.length > 0 && (
            <ul className="text-[11px] text-red-300/80 font-mono max-h-20 overflow-y-auto">
              {status.errors.slice(0, 5).map((er, i) => (
                <li key={i}>{er}</li>
              ))}
            </ul>
          )}
          {done && status.done > 0 && (onBrowse || onAnnotate) && (
            <div className="mt-1 flex gap-2">
              {onBrowse && (
                <button
                  type="button"
                  onClick={() => onBrowse(status.container_path, status.done)}
                  className="flex-1 bg-sky-600 text-white rounded-md py-2 text-sm font-medium hover:bg-sky-700 transition-colors"
                >
                  Browse this dataset
                </button>
              )}
              {onAnnotate && firstKey && (
                <button
                  type="button"
                  onClick={() => onAnnotate(status.container_path, firstKey)}
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
    </div>
  );
}
