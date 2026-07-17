import { useState, useEffect } from 'react';

const STORAGE_KEY = 'sam3_hub_selected_tab_paths';

/**
 * useHubSelectedTabs — persists the set of selected hub tab paths in localStorage.
 * Returns the stored paths (null if none) and a setter that writes through to storage.
 */
export function useHubSelectedTabs() {
  const [selectedPaths, setSelectedPathsState] = useState<string[] | null>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.every(item => typeof item === 'string')) {
          return parsed;
        }
      }
    } catch (error) {
      console.error('Error loading selected tabs from localStorage:', error);
    }
    return null;
  });

  /** Persist `paths` to localStorage and update state; logs on storage failure. */
  const setSelectedPaths = (paths: string[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(paths));
      setSelectedPathsState(paths);
    } catch (error) {
      console.error('Error saving selected tabs to localStorage:', error);
    }
  };

  return { selectedPaths, setSelectedPaths };
}
