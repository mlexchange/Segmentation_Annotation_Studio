/**
 * Denoise store — the server-side denoise setting for the currently-open
 * sample (see components/annotate/DenoisePanel).
 *
 * This used to be `useState` in AnnotatePage, which was fine while denoising
 * was purely a display concern. It isn't anymore: "Train on denoised input"
 * (TrainPage, and ApplyModelPanel's fine-tune) bakes the SAME setting into a
 * training run so inference can reapply it, and neither of those lives under
 * AnnotatePage. Lifting it here is what lets the checkbox name the exact filter
 * the user tuned in Annotate rather than offering a second, independent copy of
 * the controls that could silently disagree with what they were looking at.
 *
 * NOT persisted, unlike settingsStore/ratingStore. Those hold machine-local
 * preferences (who you are, how you like colors) that are true across sessions.
 * This is per-session view state about one open sample: `runId` names a saved
 * denoiser run that may since have been deleted, `crop` is a transient tuning
 * aid, and a good strength is a property of a particular volume's noise, not of
 * the user. Restoring it on next launch would silently re-apply a filter (each
 * slice fetch is a real server-side CPU cost) and, worse, could hand a
 * days-old setting to a training run the user believed was on raw pixels.
 */
import { create } from 'zustand';
import { NO_DENOISE, type DenoiseOpts } from '@/hooks/useImageSlice';

interface DenoiseState {
  /** Current denoise settings — `NO_DENOISE` (method `'none'`) when off. */
  denoise: DenoiseOpts;
  setDenoise: (next: DenoiseOpts) => void;
  /** Back to "off" — used by DisplayControls' "reset display options". */
  resetDenoise: () => void;
}

export const useDenoiseStore = create<DenoiseState>((set) => ({
  denoise: NO_DENOISE,
  /** Replaces the denoise settings (method, strength, and any crop/runId). */
  setDenoise: (denoise) => set({ denoise }),
  /** Turns denoising off, discarding any crop preview and run selection. */
  resetDenoise: () => set({ denoise: NO_DENOISE }),
}));
