import { useState, useEffect } from 'react';

const STORAGE_KEY = 'hub_selected_tab_paths';

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
