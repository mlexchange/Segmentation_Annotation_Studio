/**
 * ResetTiledModal — confirm-then-execute for permanently wiping every sample
 * ingested into the connected Tiled server.
 *
 * Deliberately more friction than RunsPanel's single-run delete confirm
 * (a plain `window.confirm`): this deletes everything at once and can't be
 * undone, so it gets its own dialog with a live count of what's about to be
 * lost, fetched right when the dialog opens rather than trusted from
 * whatever the page last rendered.
 */
import { useEffect, useState } from 'react';
import { Warning, X, CircleDashed, CheckCircle } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { formatApiError } from '@/lib/apiError';

export interface ResetTiledModalProps {
  serverUri: string | null;
  onClose: () => void;
  /** Called once the reset actually completes, so the page can refresh
   *  anything it has cached (e.g. Browse's in-memory listings). */
  onReset: () => void;
}

type CountState = { status: 'loading' } | { status: 'ok'; count: number } | { status: 'error'; message: string };
type DraftsCountState =
  | { status: 'loading' }
  | { status: 'ok'; drafts: number; versions: number }
  | { status: 'error'; message: string };
type ResetState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'done'; deleted: number; errors: number; draftsDeleted: number }
  | { status: 'error'; message: string };

async function fetchSampleCount(serverUri: string | null): Promise<number> {
  const params = new URLSearchParams({ limit: '2000', refresh: 'true' });
  if (serverUri) params.set('server_uri', serverUri);
  const res = await fetch(`${API_BASE}/api/browse/items?${params}`);
  if (!res.ok) throw new Error(formatApiError(await res.text(), `Request failed (${res.status}).`));
  const data = await res.json();
  return typeof data.total === 'number' ? data.total : (data.items ?? []).length;
}

async function fetchDraftsPreview(serverUri: string | null): Promise<{ drafts: number; versions: number }> {
  const params = new URLSearchParams();
  if (serverUri) params.set('server_uri', serverUri);
  const res = await fetch(`${API_BASE}/api/browse/reset-tiled/preview?${params}`);
  if (!res.ok) throw new Error(formatApiError(await res.text(), `Request failed (${res.status}).`));
  const data = await res.json();
  return { drafts: data.draft_count ?? 0, versions: data.version_count ?? 0 };
}

export default function ResetTiledModal({ serverUri, onClose, onReset }: ResetTiledModalProps) {
  const [count, setCount] = useState<CountState>({ status: 'loading' });
  const [draftsCount, setDraftsCount] = useState<DraftsCountState>({ status: 'loading' });
  const [clearDrafts, setClearDrafts] = useState(true);
  const [reset, setReset] = useState<ResetState>({ status: 'idle' });

  useEffect(() => {
    let cancelled = false;
    setCount({ status: 'loading' });
    fetchSampleCount(serverUri)
      .then((n) => { if (!cancelled) setCount({ status: 'ok', count: n }); })
      .catch((err) => {
        if (!cancelled) setCount({ status: 'error', message: err instanceof Error ? err.message : String(err) });
      });
    setDraftsCount({ status: 'loading' });
    fetchDraftsPreview(serverUri)
      .then(({ drafts, versions }) => { if (!cancelled) setDraftsCount({ status: 'ok', drafts, versions }); })
      .catch((err) => {
        if (!cancelled) setDraftsCount({ status: 'error', message: err instanceof Error ? err.message : String(err) });
      });
    return () => { cancelled = true; };
  }, [serverUri]);

  const busy = reset.status === 'running';

  const handleConfirm = async () => {
    setReset({ status: 'running' });
    try {
      const params = new URLSearchParams({ clear_drafts: String(clearDrafts) });
      if (serverUri) params.set('server_uri', serverUri);
      const res = await fetch(`${API_BASE}/api/browse/reset-tiled?${params}`, { method: 'POST' });
      if (!res.ok) {
        setReset({ status: 'error', message: formatApiError(await res.text(), `Request failed (${res.status}).`) });
        return;
      }
      const data = await res.json();
      setReset({
        status: 'done',
        deleted: Array.isArray(data.deleted_keys) ? data.deleted_keys.length : 0,
        errors: Array.isArray(data.errors) ? data.errors.length : 0,
        draftsDeleted: typeof data.drafts_deleted === 'number' ? data.drafts_deleted : 0,
      });
      onReset();
    } catch (err) {
      setReset({ status: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2 text-red-700 font-semibold">
            <Warning size={18} />
            Reset Tiled server
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="text-gray-400 hover:text-gray-700 transition-colors disabled:opacity-50"
          >
            <X size={18} />
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-4 space-y-3">
          {reset.status === 'done' ? (
            <div className="flex flex-col items-center gap-2 py-4 text-center">
              <CheckCircle size={32} className="text-emerald-600" />
              <p className="text-sm text-gray-700">
                Deleted {reset.deleted} item{reset.deleted === 1 ? '' : 's'} from Tiled.
              </p>
              {reset.draftsDeleted > 0 && (
                <p className="text-sm text-gray-700">
                  Also removed {reset.draftsDeleted} local annotation draft{reset.draftsDeleted === 1 ? '' : 's'}.
                </p>
              )}
              {reset.errors > 0 && (
                <p className="text-xs text-amber-700">
                  {reset.errors} item{reset.errors === 1 ? '' : 's'} could not be deleted — check the server logs.
                </p>
              )}
            </div>
          ) : (
            <>
              <p className="text-sm text-gray-700">
                This permanently deletes <strong>every sample ingested into this Tiled server</strong> —
                every dataset, slice, and version thumbnail. There is no undo.
              </p>
              <p className="text-sm text-gray-500">
                Your saved training runs are stored separately and are <strong>not</strong> affected.
              </p>
              <div className="rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-sm">
                {count.status === 'loading' && (
                  <span className="flex items-center gap-2 text-gray-500">
                    <CircleDashed size={14} className="animate-spin" /> Counting what's there…
                  </span>
                )}
                {count.status === 'ok' && (
                  <span className="text-gray-700">
                    <strong>{count.count}</strong> sample{count.count === 1 ? '' : 's'} will be deleted.
                  </span>
                )}
                {count.status === 'error' && (
                  <span className="text-amber-700">Could not count samples first: {count.message}</span>
                )}
              </div>
              <label className="flex items-start gap-2 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={clearDrafts}
                  onChange={(e) => setClearDrafts(e.target.checked)}
                  disabled={busy}
                  className="mt-0.5"
                />
                <span className="text-gray-700">
                  Also delete local annotation drafts/versions for this server.{' '}
                  {draftsCount.status === 'ok' && (draftsCount.drafts > 0 || draftsCount.versions > 0) && (
                    <span className="text-gray-500">
                      ({draftsCount.drafts} draft{draftsCount.drafts === 1 ? '' : 's'}, {draftsCount.versions}{' '}
                      version{draftsCount.versions === 1 ? '' : 's'})
                    </span>
                  )}
                  {draftsCount.status === 'loading' && <span className="text-gray-400">(counting…)</span>}
                  {draftsCount.status === 'error' && (
                    <span className="text-amber-700">(could not count: {draftsCount.message})</span>
                  )}
                  {!clearDrafts && (
                    <span className="block text-xs text-amber-700 mt-1">
                      Old annotations will silently reattach if a new dataset is later ingested at the same path.
                    </span>
                  )}
                </span>
              </label>
              {reset.status === 'error' && (
                <p className="text-sm text-red-600 break-words">{reset.message}</p>
              )}
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 px-5 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl">
          {reset.status === 'done' ? (
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white font-medium hover:bg-sky-700 transition-colors"
            >
              Close
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onClose}
                disabled={busy}
                className="px-4 py-2 text-sm rounded-md border border-gray-200 text-gray-700 hover:bg-gray-100 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                disabled={busy}
                className="flex items-center gap-2 px-4 py-2 text-sm rounded-md bg-red-600 text-white font-medium hover:bg-red-700 transition-colors disabled:opacity-60"
              >
                {busy ? (
                  <>
                    <CircleDashed size={16} className="animate-spin" />
                    Resetting…
                  </>
                ) : (
                  <>
                    <Warning size={16} />
                    Yes, reset Tiled
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
