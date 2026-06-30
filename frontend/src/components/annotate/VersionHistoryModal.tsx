/**
 * VersionHistoryModal — lists saved versions with annotated thumbnails and
 * Preview / Restore actions.
 */
import { useState } from 'react';
import { X, ClockCounterClockwise, ArrowCounterClockwise, Eye, Image as ImageIcon } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import type { VersionMeta } from '@/hooks/useSave';

interface Props {
  versions: VersionMeta[];
  sourceKey: string | null;
  onPreview: (version: number) => void;
  onRestore: (version: number) => void;
  onClose: () => void;
}

/** Formats an ISO timestamp as a short local date/time; falls back to the raw string. */
function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

/** Builds the backend URL for a version's rendered thumbnail. */
function thumbnailUrl(sourceKey: string, version: number): string {
  return `${API_BASE}/api/annotations/versions/${version}/thumbnail?source_key=${encodeURIComponent(sourceKey)}`;
}

/** Lazy-loading thumbnail image with a placeholder icon on load failure. */
function VersionThumbnail({ src, alt }: { src: string; alt: string }) {
  const [status, setStatus] = useState<'loading' | 'ok' | 'error'>('loading');
  return (
    <div className="w-16 h-16 flex-shrink-0 rounded-md overflow-hidden bg-gray-100 flex items-center justify-center border border-gray-200">
      {status === 'error' ? (
        <ImageIcon size={22} className="text-gray-300" />
      ) : (
        <img
          src={src}
          alt={alt}
          className={`w-full h-full object-cover transition-opacity duration-200 ${status === 'ok' ? 'opacity-100' : 'opacity-0'}`}
          onLoad={() => setStatus('ok')}
          onError={() => setStatus('error')}
        />
      )}
    </div>
  );
}

/** Renders the modal listing saved versions with preview/restore actions. */
export default function VersionHistoryModal({ versions, sourceKey, onPreview, onRestore, onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-xl shadow-2xl w-[480px] max-h-[75vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2 text-sky-900 font-semibold">
            <ClockCounterClockwise size={18} />
            Version History
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Version list */}
        <div className="overflow-y-auto flex-1 px-4 py-3 space-y-2">
          {versions.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-8">
              No saved versions yet. Click Save to create the first one.
            </p>
          ) : (
            [...versions].reverse().map((v) => (
              <div
                key={v.version}
                className="flex items-center gap-3 rounded-lg border border-gray-200 px-3 py-2.5 hover:border-sky-300 hover:bg-sky-50 transition-colors group"
              >
                {/* Thumbnail */}
                {sourceKey && (
                  <VersionThumbnail
                    src={thumbnailUrl(sourceKey, v.version)}
                    alt={`Version ${v.version} thumbnail`}
                  />
                )}

                {/* Metadata */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-800">
                    v{v.version}
                    <span className="ml-2 text-xs text-gray-500">{formatDate(v.saved_at)}</span>
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    {v.shape_count} shape{v.shape_count !== 1 ? 's' : ''} &middot;{' '}
                    {v.class_count} class{v.class_count !== 1 ? 'es' : ''}
                    {v.annotated_by && (
                      <> &middot; by {v.annotated_by}</>
                    )}
                  </div>
                  {v.notes && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-2" title={v.notes}>
                      {v.notes}
                    </p>
                  )}
                </div>

                {/* Actions (visible on row hover) */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                  <button
                    type="button"
                    onClick={() => onPreview(v.version)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium text-gray-600 hover:bg-gray-100 transition-colors"
                    title={`Preview version ${v.version} on canvas`}
                  >
                    <Eye size={14} />
                    Preview
                  </button>
                  <button
                    type="button"
                    onClick={() => { onRestore(v.version); onClose(); }}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium text-sky-700 hover:bg-sky-100 transition-colors"
                    title={`Restore version ${v.version}`}
                  >
                    <ArrowCounterClockwise size={14} />
                    Restore
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-200">
          <p className="text-xs text-gray-400">
            Preview scrubs versions on the canvas without changing your work. Restoring loads
            the selected version into the editor so you can save it as a new version.
          </p>
        </div>
      </div>
    </div>
  );
}
