/**
 * ConflictDialog — shown when a pre-flight check finds that some of the images
 * about to be uploaded already exist in the destination dataset.
 *
 * Appears BEFORE any bytes leave the browser, and lets the user pick how to
 * resolve the collision rather than watching every file fail with a 409.
 */
import { ArrowRight, Copy, SkipForward, Trash, X } from '@phosphor-icons/react';

export interface IngestConflict {
  filename: string;
  key: string;
}

interface ConflictDialogProps {
  /** Destination container the collision was detected in. */
  containerPath: string;
  /** Number of supported files in this upload. */
  totalFiles: number;
  conflicts: IngestConflict[];
  /** Total children already in the destination container. */
  existingCount: number;
  /** Unused sibling name to offer as a fresh destination. */
  suggestedContainerPath: string;
  onReplace: () => void;
  onSkip: () => void;
  onNewDataset: () => void;
  onBrowseExisting: () => void;
  onCancel: () => void;
}

const MAX_LISTED = 5;

export default function ConflictDialog({
  containerPath,
  totalFiles,
  conflicts,
  existingCount,
  suggestedContainerPath,
  onReplace,
  onSkip,
  onNewDataset,
  onBrowseExisting,
  onCancel,
}: ConflictDialogProps) {
  const allConflict = conflicts.length >= totalFiles;
  const newCount = totalFiles - conflicts.length;

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 p-4"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-lg overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-700 px-4 py-2.5">
          <span className="text-sm font-semibold text-white">
            This dataset already has these images
          </span>
          <button
            type="button"
            onClick={onCancel}
            className="text-slate-400 hover:text-white transition-colors"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="space-y-3 px-4 py-3">
          <p className="text-sm text-sky-100">
            {conflicts.length} of {totalFiles} image{totalFiles === 1 ? '' : 's'}{' '}
            {conflicts.length === 1 ? 'is' : 'are'} already in{' '}
            <span className="font-mono text-sky-300">{containerPath}</span>
            {existingCount > 0 && (
              <span className="text-sky-300/80">
                {' '}
                ({existingCount} sample{existingCount === 1 ? '' : 's'} there now)
              </span>
            )}
            .
          </p>

          <ul className="max-h-24 overflow-y-auto rounded-md bg-white/5 px-3 py-2 text-[11px] font-mono text-sky-200/90">
            {conflicts.slice(0, MAX_LISTED).map((c) => (
              <li key={c.key}>{c.filename}</li>
            ))}
            {conflicts.length > MAX_LISTED && (
              <li className="text-sky-300/70">
                …and {conflicts.length - MAX_LISTED} more
              </li>
            )}
          </ul>

          <div className="space-y-2">
            <button
              type="button"
              onClick={onReplace}
              className="flex w-full items-center gap-2 rounded-md bg-sky-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-700"
            >
              <Trash size={15} />
              <span className="flex-1 text-left">
                Replace the existing image{conflicts.length === 1 ? '' : 's'}
                <span className="block text-[11px] font-normal text-sky-100/80">
                  Overwrites {conflicts.length} sample
                  {conflicts.length === 1 ? '' : 's'} in this dataset
                </span>
              </span>
            </button>

            <button
              type="button"
              onClick={onSkip}
              disabled={allConflict}
              className="flex w-full items-center gap-2 rounded-md border border-white/20 bg-white/5 px-3 py-2 text-sm font-medium text-sky-100 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <SkipForward size={15} />
              <span className="flex-1 text-left">
                Skip the duplicates
                <span className="block text-[11px] font-normal text-sky-300/80">
                  {allConflict
                    ? 'Nothing new to add — every image is already here'
                    : `Ingests only the ${newCount} new image${newCount === 1 ? '' : 's'}`}
                </span>
              </span>
            </button>

            <button
              type="button"
              onClick={onNewDataset}
              className="flex w-full items-center gap-2 rounded-md border border-white/20 bg-white/5 px-3 py-2 text-sm font-medium text-sky-100 transition-colors hover:bg-white/10"
            >
              <Copy size={15} />
              <span className="flex-1 text-left">
                Ingest to a new dataset
                <span className="block text-[11px] font-normal font-mono text-sky-300/80">
                  {suggestedContainerPath}
                </span>
              </span>
            </button>

            <button
              type="button"
              onClick={onBrowseExisting}
              className="flex w-full items-center gap-2 rounded-md border border-white/20 bg-white/5 px-3 py-2 text-sm font-medium text-sky-100 transition-colors hover:bg-white/10"
            >
              <ArrowRight size={15} />
              <span className="flex-1 text-left">
                Browse the existing dataset
                <span className="block text-[11px] font-normal text-sky-300/80">
                  Cancels this upload and opens it in Browse
                </span>
              </span>
            </button>
          </div>

          <button
            type="button"
            onClick={onCancel}
            className="w-full py-1 text-xs text-sky-300/80 transition-colors hover:text-sky-100"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
