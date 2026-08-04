/**
 * FeedbackConsentModal — shows exactly what will be sent to the external
 * feedback form before any network request is made.
 *
 * The feedback context includes the currently-open dataset's source path,
 * which can carry usernames, study identifiers, or project names — real
 * information the user should see and can decline to send, rather than
 * discovering only after the form iframe has already loaded it.
 */
import { WarningCircle } from '@phosphor-icons/react';

interface FeedbackConsentModalProps {
  context: string;
  onSend: () => void;
  onCancel: () => void;
}

export default function FeedbackConsentModal({ context, onSend, onCancel }: FeedbackConsentModalProps) {
  return (
    <div
      className="fixed inset-0 z-[10001] flex items-center justify-center bg-black/50 p-4"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-labelledby="feedback-consent-title"
    >
      <div
        className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-slate-700 px-4 py-3">
          <span id="feedback-consent-title" className="text-sm font-semibold text-white">
            Send this to the feedback form?
          </span>
        </div>
        <div className="flex-1 overflow-auto px-4 py-3">
          <p className="mb-2 flex items-start gap-2 text-xs text-amber-300">
            <WarningCircle size={16} className="mt-0.5 shrink-0" />
            This leaves your machine and goes to the external form below. It may include
            the currently open sample's path, which can contain identifying names.
          </p>
          <pre className="whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs text-slate-200">
            {context}
          </pre>
        </div>
        <div className="flex items-center justify-end gap-3 border-t border-slate-700 px-4 py-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md px-3 py-1.5 text-sm text-slate-300 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSend}
            className="rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-700 transition-colors"
          >
            Send Feedback
          </button>
        </div>
      </div>
    </div>
  );
}
