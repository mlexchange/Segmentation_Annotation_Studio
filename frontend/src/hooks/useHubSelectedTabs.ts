import { useState } from 'react';

const STORAGE_KEY = 'sam3_hub_selected_tab_paths';
// The full route-path universe as of the last save. Lets App.tsx tell "a route
// the user explicitly hid" apart from "a route added by an app update the user
// has never had a chance to opt out of" — both look identical as a plain diff
// against the current route list, so that distinction has to be persisted.
const KNOWN_KEY = 'sam3_hub_known_tab_paths';

function readPathArray(key: string): string[] | null {
  try {
    const stored = localStorage.getItem(key);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === 'string')) {
        return parsed;
      }
    }
  } catch (error) {
    console.error(`Error loading ${key} from localStorage:`, error);
  }
  return null;
}

/**
 * useHubSelectedTabs — persists the set of selected hub tab paths (and the
 * route universe they were chosen from) in localStorage.
 *
 * Returns the stored paths (null if none) and a setter that writes through to
 * storage. In-memory state always updates even if the storage write fails —
 * a private-browsing/quota error should degrade to "preference resets next
 * launch," never to a permanently blank app (see App.tsx's `needsInit` gate).
 */
export function useHubSelectedTabs() {
  const [selectedPaths, setSelectedPathsState] = useState<string[] | null>(() => readPathArray(STORAGE_KEY));
  const [knownPaths, setKnownPathsState] = useState<string[] | null>(() => readPathArray(KNOWN_KEY));

  /** Persist `paths` (and optionally the route universe `known`) and update state. */
  const setSelectedPaths = (paths: string[], known?: string[]) => {
    setSelectedPathsState(paths);
    if (known) setKnownPathsState(known);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(paths));
      if (known) localStorage.setItem(KNOWN_KEY, JSON.stringify(known));
    } catch (error) {
      console.error('Error saving selected tabs to localStorage (using in-memory state for this session):', error);
    }
  };

  return { selectedPaths, knownPaths, setSelectedPaths };
}
