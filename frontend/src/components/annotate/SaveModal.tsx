/**
 * SaveModal — confirm save with annotator name, notes, and thumbnail preview.
 */
import { useEffect, useRef, useState } from 'react';
import { FloppyDisk, X, CircleDashed, Image as ImageIcon } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import type { SaveDraftPayload } from '@/hooks/useSave';

const ANNOTATOR_STORAGE_KEY = 'sam3_annotator_name';

export interface SaveModalProps {
  sourceKey: string;
  payload: SaveDraftPayload;
  shapeCount: number;
  classCount: number;
  isSaving: boolean;
  onSave: (opts: { annotatedBy: string; notes: string; thumbnailBase64?: string }) => Promise<void>;
  onClose: () => void;
}

/** POSTs the draft to the backend to render a preview thumbnail; returns null on failure. */
async function fetchPreviewBlob(sourceKey: string, payload: SaveDraftPayload): Promise<Blob | null> {
  try {
    const res = await fetch(
      `${API_BASE}/api/annotations/preview-thumbnail?source_key=${encodeURIComponent(sourceKey)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );
    if (!res.ok) return null;
    return await res.blob();
  } catch {
    return null;
  }
}

/** Reads a Blob into a base64 string (without the data-URL prefix). */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') {
        reject(new Error('Failed to read preview blob'));
        return;
      }
      resolve(result.split(',')[1] ?? '');
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read preview blob'));
    reader.readAsDataURL(blob);
  });
}

/** Renders the save-version dialog and fetches a live thumbnail preview of the draft. */
export default function SaveModal({
  sourceKey,
  payload,
  shapeCount,
  classCount,
  isSaving,
  onSave,
  onClose,
}: SaveModalProps) {
  const [annotatedBy, setAnnotatedBy] = useState(() => {
    try {
      return localStorage.getItem(ANNOTATOR_STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  });
  const [notes, setNotes] = useState('');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(true);
  const previewBlobRef = useRef<Blob | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setPreviewLoading(true);
    setPreviewUrl(null);
    previewBlobRef.current = null;

    fetchPreviewBlob(sourceKey, payload).then((blob) => {
      if (cancelled) return;
      if (blob) {
        previewBlobRef.current = blob;
        objectUrl = URL.createObjectURL(blob);
        setPreviewUrl(objectUrl);
      }
      setPreviewLoading(false);
    });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [sourceKey, payload]);

  /** Persists the annotator name to localStorage, encodes the thumbnail, and invokes onSave. */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      localStorage.setItem(ANNOTATOR_STORAGE_KEY, annotatedBy.trim());
    } catch {
      // ignore storage errors
    }
    let thumbnailBase64: string | undefined;
    if (previewBlobRef.current) {
      try {
        thumbnailBase64 = await blobToBase64(previewBlobRef.current);
      } catch {
        // fall back to server-side render on save
      }
    }
    await onSave({
      annotatedBy: annotatedBy.trim(),
      notes: notes.trim(),
      thumbnailBase64,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2 text-sky-900 font-semibold">
            <FloppyDisk size={18} />
            Save version
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isSaving}
            className="text-gray-400 hover:text-gray-700 transition-colors disabled:opacity-50"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col overflow-hidden">
          <div className="overflow-y-auto px-5 py-4 space-y-4">
            {/* Thumbnail preview */}
            <div>
              <label className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
                Preview
              </label>
              <div className="mt-1.5 rounded-lg border border-gray-200 bg-gray-50 overflow-hidden flex items-center justify-center min-h-[160px]">
                {previewLoading ? (
                  <div className="flex items-center gap-2 text-sm text-gray-400 py-8">
                    <CircleDashed size={18} className="animate-spin" />
                    Generating preview…
                  </div>
                ) : previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Annotation preview"
                    className="max-h-48 w-full object-contain"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-1 text-gray-400 py-8">
                    <ImageIcon size={28} />
                    <span className="text-xs">Preview unavailable</span>
                  </div>
                )}
              </div>
              <p className="mt-1 text-xs text-gray-400">
                {shapeCount} shape{shapeCount !== 1 ? 's' : ''} · {classCount} class{classCount !== 1 ? 'es' : ''}
              </p>
            </div>

            {/* Annotator */}
            <div>
              <label htmlFor="annotated-by" className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
                Who annotated this
              </label>
              <input
                id="annotated-by"
                type="text"
                value={annotatedBy}
                onChange={(e) => setAnnotatedBy(e.target.value)}
                placeholder="Your name or initials"
                className="mt-1.5 w-full border border-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                autoComplete="name"
              />
            </div>

            {/* Notes */}
            <div>
              <label htmlFor="save-notes" className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
                Notes
              </label>
              <textarea
                id="save-notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional notes about this version…"
                rows={3}
                className="mt-1.5 w-full border border-gray-200 rounded-md px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 px-5 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl">
            <button
              type="button"
              onClick={onClose}
              disabled={isSaving}
              className="px-4 py-2 text-sm rounded-md border border-gray-200 text-gray-700 hover:bg-gray-100 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSaving}
              className="flex items-center gap-2 px-4 py-2 text-sm rounded-md bg-sky-600 text-white font-medium hover:bg-sky-700 transition-colors disabled:opacity-60"
            >
              {isSaving ? (
                <>
                  <CircleDashed size={16} className="animate-spin" />
                  Saving…
                </>
              ) : (
                <>
                  <FloppyDisk size={16} />
                  Save version
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
