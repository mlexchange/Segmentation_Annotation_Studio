/**
 * Settings store — machine-local user preferences, persisted to localStorage.
 *
 * `annotatorName` identifies who is doing the annotating. There is no login;
 * this name defaults into saved-version metadata and is stamped into COCO
 * exports so downloads are self-identifying for external inter-annotator
 * agreement analysis (each annotator runs their own local install).
 *
 * `colorblindMode` switches new annotation-class colors to a colorblind-safe
 * palette (see lib/classColors). It is a display/authoring preference only and
 * does not change any exported data format.
 *
 * `sessionId` is a persisted, anonymous install identifier (no PII) — generated
 * once and stamped into bug reports so feedback is self-identifying per install.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/** Random id, preferring crypto.randomUUID with a plain fallback. */
function genId(): string {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  } catch { /* fall through */ }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

interface SettingsStore {
  annotatorName: string;
  setAnnotatorName: (name: string) => void;
  colorblindMode: boolean;
  setColorblindMode: (on: boolean) => void;
  /** Anonymous install id (persisted; seeded once). */
  sessionId: string;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      annotatorName: '',
      setAnnotatorName: (name) => set({ annotatorName: name }),
      colorblindMode: false,
      setColorblindMode: (on) => set({ colorblindMode: on }),
      sessionId: genId(), // overridden by the persisted value after first run
    }),
    { name: 'sam3_settings' },
  ),
);
