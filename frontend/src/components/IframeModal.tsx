/**
 * IframeModal — embeds an external page (Docs, Feedback form) in an in-app overlay
 * instead of opening a new browser tab. Includes an "Open in new tab" fallback for
 * targets that refuse to be framed (X-Frame-Options / frame-ancestors CSP).
 */
import { ArrowSquareOut, X } from '@phosphor-icons/react';

interface IframeModalProps {
  title: string;
  url: string;
  onClose: () => void;
}

export default function IframeModal({ title, url, onClose }: IframeModalProps) {
  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-700 px-4 py-2.5">
          <span className="text-sm font-semibold text-white">{title}</span>
          <div className="flex items-center gap-3">
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-sky-300 hover:text-sky-100 transition-colors"
              title="Open in a new tab"
            >
              <ArrowSquareOut size={14} /> Open in new tab
            </a>
            <button
              type="button"
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>
        <iframe
          src={url}
          title={title}
          className="min-h-0 w-full flex-1 bg-white"
          // Allow the embedded page's own scripts/forms; sandbox is intentionally
          // permissive since these are first-party/trusted (docs + our feedback form).
        />
      </div>
    </div>
  );
}
