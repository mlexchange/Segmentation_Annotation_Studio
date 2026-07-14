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
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsStore {
  annotatorName: string;
  setAnnotatorName: (name: string) => void;
  colorblindMode: boolean;
  setColorblindMode: (on: boolean) => void;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      annotatorName: '',
      setAnnotatorName: (name) => set({ annotatorName: name }),
      colorblindMode: false,
      setColorblindMode: (on) => set({ colorblindMode: on }),
    }),
    { name: 'sam3_settings' },
  ),
);
