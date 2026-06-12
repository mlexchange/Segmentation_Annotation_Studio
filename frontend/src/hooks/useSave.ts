/**
 * useSave — explicit versioned save + version history for one sourceKey.
 *
 * Separate from useDraftSync (which handles crash-recovery autosave).
 * Tracks dirty state: isDirty becomes true whenever store data changes
 * after the last explicit save.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from '@/config';
import { useAnnotationStore, type Shape } from '@/stores/annotationStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';

export interface VersionMeta {
  version: number;
  saved_at: string;
  shape_count: number;
  class_count: number;
  annotated_by?: string | null;
  notes?: string | null;
  has_thumbnail?: boolean;
}

/** Draft payload sent to save / preview endpoints. */
export interface SaveDraftPayload {
  classes: AnnotationClass[];
  slices: Record<string, Shape[]>;
  split_by_slice: Record<string, string>;
  negative_slices: string[];
}

/** Full annotation payload for one saved version (used for preview + restore). */
export interface VersionPayload extends SaveDraftPayload {}

export interface SaveOptions {
  annotatedBy?: string;
  notes?: string;
  /** PNG from the save-modal preview — skips server-side re-render. */
  thumbnailBase64?: string;
}

export interface UseSaveReturn {
  /** True when there are unsaved changes since the last explicit save. */
  isDirty: boolean;
  /** True while a save request is in flight. */
  isSaving: boolean;
  /** ISO-8601 timestamp of the last successful explicit save, or null. */
  lastSavedAt: string | null;
  /** Build the current draft payload for preview / save. */
  buildSavePayload: () => SaveDraftPayload | null;
  /** Shape and class counts for the current source. */
  saveSummary: { shapeCount: number; classCount: number };
  /** Call to create a new version and sync Tiled metadata. */
  save: (options?: SaveOptions) => Promise<boolean>;
  /** Cached list of versions (refreshed after each save). */
  versions: VersionMeta[];
  /** Refresh the version list from the server. */
  refreshVersions: () => Promise<void>;
  /** Fetch a version's full payload (cached) — used for non-destructive preview. */
  fetchVersionPayload: (version: number) => Promise<VersionPayload | null>;
  /** Load a specific version's payload back into the stores. */
  restoreVersion: (version: number) => Promise<void>;
  /** Mark the current state as clean (call after draft restore on open). */
  markClean: () => void;
}

export function useSave(sourceKey: string | null): UseSaveReturn {
  const byImage = useAnnotationStore((s) => s.byImage);
  const splitBySlice = useAnnotationStore((s) => s.splitBySlice);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const mergeSourceDraft = useAnnotationStore((s) => s.mergeSourceDraft);
  const classes = useClassStore((s) => s.classes);
  const setClasses = useClassStore((s) => s.setClasses);

  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState<string | null>(null);
  const [versions, setVersions] = useState<VersionMeta[]>([]);

  const cleanRef = useRef(true);
  const payloadCacheRef = useRef<Map<number, VersionPayload>>(new Map());

  useEffect(() => {
    if (cleanRef.current) {
      cleanRef.current = false;
      return;
    }
    setIsDirty(true);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [byImage, splitBySlice, negativeSlices, classes]);

  useEffect(() => {
    cleanRef.current = true;
    setIsDirty(false);
    setLastSavedAt(null);
    setVersions([]);
    payloadCacheRef.current.clear();
  }, [sourceKey]);

  const buildSavePayload = useCallback((): SaveDraftPayload | null => {
    if (!sourceKey) return null;
    return {
      classes,
      slices: byImage[sourceKey] ?? {},
      split_by_slice: splitBySlice[sourceKey] ?? {},
      negative_slices: negativeSlices[sourceKey] ?? [],
    };
  }, [sourceKey, byImage, splitBySlice, negativeSlices, classes]);

  const saveSummary = useMemo(() => {
    const payload = buildSavePayload();
    if (!payload) return { shapeCount: 0, classCount: classes.length };
    const shapeCount = Object.values(payload.slices).reduce((n, shapes) => n + shapes.length, 0);
    return { shapeCount, classCount: payload.classes.length };
  }, [buildSavePayload, classes.length]);

  const refreshVersions = useCallback(async () => {
    if (!sourceKey) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/annotations/versions?source_key=${encodeURIComponent(sourceKey)}`
      );
      if (!res.ok) return;
      const data: VersionMeta[] = await res.json();
      setVersions(data);
    } catch {
      // non-fatal
    }
  }, [sourceKey]);

  const save = useCallback(
    async (options: SaveOptions = {}): Promise<boolean> => {
      if (!sourceKey) return false;
      const payload = buildSavePayload();
      if (!payload) return false;

      setIsSaving(true);
      try {
        const res = await fetch(
          `${API_BASE}/api/annotations/save?source_key=${encodeURIComponent(sourceKey)}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              payload,
              annotated_by: options.annotatedBy ?? '',
              notes: options.notes ?? '',
              thumbnail_base64: options.thumbnailBase64 ?? null,
            }),
          }
        );
        if (!res.ok) throw new Error(`Save failed: ${res.status}`);
        const data = await res.json();
        setLastSavedAt(data.saved_at as string);
        setIsDirty(false);
        cleanRef.current = false;
        await refreshVersions();
        return true;
      } catch (err) {
        console.error('Save failed:', err);
        return false;
      } finally {
        setIsSaving(false);
      }
    },
    [sourceKey, buildSavePayload, refreshVersions]
  );

  const fetchVersionPayload = useCallback(
    async (version: number): Promise<VersionPayload | null> => {
      if (!sourceKey) return null;
      const cached = payloadCacheRef.current.get(version);
      if (cached) return cached;
      try {
        const res = await fetch(
          `${API_BASE}/api/annotations/versions/${version}?source_key=${encodeURIComponent(sourceKey)}`
        );
        if (!res.ok) throw new Error(`Version fetch failed: ${res.status}`);
        const doc = await res.json();
        const raw = (doc.payload ?? {}) as Record<string, unknown>;
        const payload: VersionPayload = {
          classes: Array.isArray(raw.classes) ? (raw.classes as AnnotationClass[]) : [],
          slices: (raw.slices ?? {}) as Record<string, Shape[]>,
          split_by_slice: (raw.split_by_slice ?? {}) as Record<string, string>,
          negative_slices: (raw.negative_slices ?? []) as string[],
        };
        payloadCacheRef.current.set(version, payload);
        return payload;
      } catch (err) {
        console.error('Version fetch failed:', err);
        return null;
      }
    },
    [sourceKey]
  );

  const restoreVersion = useCallback(
    async (version: number) => {
      if (!sourceKey) return;
      const payload = await fetchVersionPayload(version);
      if (!payload) return;
      setClasses(payload.classes);
      mergeSourceDraft(sourceKey, payload.slices, payload.split_by_slice, payload.negative_slices);
      cleanRef.current = false;
      setIsDirty(true);
    },
    [sourceKey, setClasses, mergeSourceDraft, fetchVersionPayload]
  );

  const markClean = useCallback(() => {
    cleanRef.current = true;
    setIsDirty(false);
  }, []);

  useEffect(() => {
    refreshVersions();
  }, [refreshVersions]);

  return {
    isDirty,
    isSaving,
    lastSavedAt,
    buildSavePayload,
    saveSummary,
    save,
    versions,
    refreshVersions,
    fetchVersionPayload,
    restoreVersion,
    markClean,
  };
}
