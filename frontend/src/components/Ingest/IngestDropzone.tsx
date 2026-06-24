/**
 * IngestDropzone — drag-and-drop (or pick) image files/folders and copy them
 * into the connected Tiled server as individually-browsable samples.
 *
 * Shown on the Connect page once a Tiled server is selected. On success it can
 * point the Browse view at the freshly-created container.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { UploadSimple, Warning, CheckCircle } from '@phosphor-icons/react';
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
}

function isSupported(name: string): boolean {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  return SUPPORTED_EXTS.includes(ext);
}

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

export default function IngestDropzone({ serverUri, onBrowse }: IngestDropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const [containerPath, setContainerPath] = useState('browse/');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

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

  const startUpload = useCallback(
    async (files: File[], suggestedName: string) => {
      const supported = files.filter((f) => isSupported(f.name));
      if (supported.length === 0) {
        setError('No supported image files (TIFF, PNG, JPG, NPY).');
        return;
      }
      // Default the container name from the dropped folder if still unset.
      let target = containerPath.trim();
      if (!target || target === 'browse/') {
        target = `browse/${sanitizeName(suggestedName || 'dataset')}`;
        setContainerPath(target);
      }

      setError(null);
      setStatus(null);
      setUploading(true);
      try {
        const fd = new FormData();
        fd.append('container_path', target);
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
    [containerPath, serverUri, poll],
  );

  const inputRef = useRef<HTMLInputElement | null>(null);

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

      <div
        role="button"
        tabIndex={0}
        onClick={() => !uploading && inputRef.current?.click()}
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
        <input
          ref={inputRef}
          type="file"
          multiple
          // @ts-expect-error — non-standard but widely supported for folder selection
          webkitdirectory=""
          onChange={onPick}
          disabled={uploading}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-2 pointer-events-none">
          <UploadSimple size={28} className="text-sky-300" />
          <p className="text-sm text-sky-100 font-medium">
            {uploading ? 'Uploading…' : 'Drag a folder of images here'}
          </p>
          <p className="text-xs text-sky-300/70">or click to choose a folder (TIFF, PNG, JPG, NPY)</p>
        </div>
      </div>

      <label className="block text-xs text-sky-300/80">
        Target container
        <input
          type="text"
          value={containerPath}
          onChange={(e) => setContainerPath(e.target.value)}
          placeholder="browse/my_dataset"
          className="mt-1 w-full border border-white/20 rounded-md px-2 py-1.5 text-sm font-mono bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
        />
      </label>

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
          {done && status.done > 0 && onBrowse && (
            <button
              type="button"
              onClick={() => onBrowse(status.container_path, status.done)}
              className="mt-1 w-full bg-sky-600 text-white rounded-md py-2 text-sm font-medium hover:bg-sky-700 transition-colors"
            >
              Browse this dataset
            </button>
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
