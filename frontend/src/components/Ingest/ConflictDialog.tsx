/**
 * ConflictDialog — shown before uploading into a destination that already has
 * content, so the user chooses what happens instead of finding out afterwards.
 *
 * Two situations, distinguished by whether `conflicts` is empty:
 *
 * 1. **Duplicate filenames** — some uploads would overwrite existing images.
 *    Replace / skip / new dataset / browse.
 * 2. **Already-populated dataset, no duplicates** — the destination is an
 *    existing sample and these files would be ADDED to it. Uploading a second
 *    dataset with different filenames lands here, and it needs asking about just
 *    as much: merging grows the earlier sample rather than making a new one, and
 *    since a sample's slice order follows its children's lexical key order, new
 *    files can renumber the slices existing annotations refer to. So a new
 *    dataset is the recommended action and merging is an explicit opt-in.
 *
 * Appears BEFORE any bytes leave the browser.
 */
import { ArrowRight, Copy, Plus, SkipForward, Trash, X } from '@phosphor-icons/react';

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
  /** How the current grouping choice would split this batch into samples. */
  samplePreview?: { sample: string; n_images: number }[];
  onReplace: () => void;
  onSkip: () => void;
  /** Merge into the existing dataset. Only offered in the no-duplicates case,
   *  where it's a real (if rarely-wanted) choice rather than an overwrite. */
  onAddToExisting: () => void;
  onNewDataset: () => void;
  onBrowseExisting: () => void;
  onCancel: () => void;
}

const MAX_LISTED = 5;
const MAX_PREVIEW = 5;

export default function ConflictDialog({
  containerPath,
  totalFiles,
  conflicts,
  existingCount,
  suggestedContainerPath,
  samplePreview,
  onReplace,
  onSkip,
  onAddToExisting,
  onNewDataset,
  onBrowseExisting,
  onCancel,
}: ConflictDialogProps) {
  const allConflict = conflicts.length >= totalFiles;
  const newCount = totalFiles - conflicts.length;
  // No filename clashes, but the destination already holds slices — this upload
  // would grow that existing sample. See the module docstring.
  const isMergeOnly = conflicts.length === 0;

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
            {isMergeOnly
              ? 'This dataset already has images in it'
              : 'This dataset already has these images'}
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
          {isMergeOnly ? (
            <>
              <p className="text-sm text-sky-100">
                <span className="font-mono text-sky-300">{containerPath}</span> already holds{' '}
                {existingCount} image{existingCount === 1 ? '' : 's'}. Uploading these{' '}
                {totalFiles} would <strong>add to that same dataset</strong>, not create a new one.
              </p>
              <p className="text-xs text-amber-300/90">
                None of the filenames clash, so nothing would be overwritten — but the dataset
                would grow to {existingCount + totalFiles} slices, and because slice order follows
                filename order, the new images can land between the existing ones and renumber
                the slices any current annotations point at.
              </p>
            </>
          ) : (
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
          )}

          {samplePreview && samplePreview.length > 0 && (
            <p className="text-xs text-sky-300/80">
              With the current grouping, this upload will create {samplePreview.length} sample
              {samplePreview.length === 1 ? '' : 's'}:{' '}
              {samplePreview
                .slice(0, MAX_PREVIEW)
                .map((s) => `${s.sample || '(this whole batch)'} (${s.n_images})`)
                .join(', ')}
              {samplePreview.length > MAX_PREVIEW && `, …and ${samplePreview.length - MAX_PREVIEW} more`}
            </p>
          )}

          {!isMergeOnly && (
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
          )}

          <div className="space-y-2">
            {/* New dataset first and highlighted when merging is the only risk —
                it's what "I'm uploading my next dataset" almost always means. */}
            <button
              type="button"
              onClick={onNewDataset}
              className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                isMergeOnly
                  ? 'bg-sky-600 text-white hover:bg-sky-700'
                  : 'border border-white/20 bg-white/5 text-sky-100 hover:bg-white/10'
              }`}
            >
              <Copy size={15} />
              <span className="flex-1 text-left">
                Ingest to a new dataset
                <span
                  className={`block text-[11px] font-normal font-mono ${
                    isMergeOnly ? 'text-sky-100/80' : 'text-sky-300/80'
                  }`}
                >
                  {suggestedContainerPath}
                </span>
              </span>
            </button>

            {isMergeOnly ? (
              <button
                type="button"
                onClick={onAddToExisting}
                className="flex w-full items-center gap-2 rounded-md border border-white/20 bg-white/5 px-3 py-2 text-sm font-medium text-sky-100 transition-colors hover:bg-white/10"
              >
                <Plus size={15} />
                <span className="flex-1 text-left">
                  Add to the existing dataset
                  <span className="block text-[11px] font-normal text-sky-300/80">
                    Makes it one {existingCount + totalFiles}-slice dataset
                  </span>
                </span>
              </button>
            ) : (
              <>
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
              </>
            )}

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
