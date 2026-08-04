import { useState, useEffect } from 'react';
import { X, Image as ImageIcon, PencilSimple, Brain } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import type { BrowseItem } from './hooks/useBrowseData';

interface BrowseDetailPanelProps {
  item: BrowseItem;
  onClose: () => void;
  serverUri?: string;
  onOpenInAnnotate?: () => void;
  /** Omitted (or the caller passes undefined) when this sample has no saved
   *  annotation yet — there is nothing to fine-tune on. */
  onOpenInTrain?: () => void;
  /** Panel width in px; the preview image scales with it. */
  width?: number;
}

const SECTION_ORDER = [
  { label: 'Annotation', keys: ['studio_annotated', 'studio_shape_count', 'studio_class_count', 'studio_updated_at'] },
  { label: 'Identity', keys: ['ThinFilmID', 'sample_name', 'PS_ID', 'BatchID', 'bar', 'SampleDescription'] },
  { label: 'Experiment', keys: ['PI', 'beamline', 'technique', 'scan_type', 'date', 'scan_date', 'energy_keV', 'incident_angle_deg', 'Exposure time s'] },
  { label: 'Geometry', keys: ['beam_x', 'beam_y', 'sdd_mm', 'pixel_size_x', 'pixel_size_y', 'sample_detector_distance'] },
  { label: 'Thin Film', keys: ['Substrate', 'SpinSpeed', 'SpinDuration', 'SpinAtmosphere', 'AnnealingTemp', 'Temp', 'AnnealingDuration', 'AnnealingAtmosphere', 'StorageLocation', 'SubstratePrep', 'PMMA-Coating'] },
  { label: 'Chemistry', keys: ['OrganicSalt', 'OrganicSalt_Abbrev', 'MetalSalt', 'MetalSalt_Abbrev', 'Solvent', 'Concentration_M', 'MixingRatio', 'TargetStoichiometry', 'Stoichiometry'] },
];

/** Render a metadata value for display: arrays/objects are summarised and ISO dates are localised. */
function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (Array.isArray(v)) return `[${v.length} items]`;
  if (typeof v === 'object') return JSON.stringify(v).slice(0, 80);
  const text = String(v);
  if (/^\d{4}-\d{2}-\d{2}T/.test(text)) {
    const parsed = Date.parse(text);
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toLocaleString(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    }
  }
  return text;
}

/** Map a raw metadata key to a friendlier label, falling back to the key itself. */
function displayKey(key: string): string {
  const aliases: Record<string, string> = {
    studio_annotated: 'Annotated',
    studio_shape_count: 'Shape count',
    studio_class_count: 'Class count',
    studio_updated_at: 'Annotated at',
  };
  return aliases[key] ?? key;
}

/**
 * BrowseDetailPanel — side panel showing a sample's thumbnail and grouped metadata.
 * Fetches a backend thumbnail (size quantised to the panel width) and offers
 * Open in Annotate / Open in Train.
 */
export default function BrowseDetailPanel({
  item,
  onClose,
  serverUri,
  onOpenInAnnotate,
  onOpenInTrain,
  width = 340,
}: BrowseDetailPanelProps) {
  const meta = item.metadata;

  // Preview scales with the panel width. The fetched thumbnail size is
  // quantised to 128px steps so resizing doesn't spam the backend.
  const previewHeight = Math.round(Math.min(Math.max(width * 0.9, 180), 760));
  // Backend caps thumbnail size at 512; request in 128px steps to limit refetches.
  const requestedSize = Math.min(512, Math.max(256, Math.ceil(width / 128) * 128));

  // Thumbnail state
  const [thumbSrc, setThumbSrc] = useState<string | null>(null);
  const [thumbStatus, setThumbStatus] = useState<'loading' | 'ok' | 'error'>('loading');

  useEffect(() => {
    setThumbSrc(null);
    setThumbStatus('loading');
    const params = new URLSearchParams({ tiled_path: item.path, size: String(requestedSize) });
    if (serverUri) params.set('server_uri', serverUri);
    const url = `${API_BASE}/api/browse/thumbnail?${params}`;
    const img = new window.Image();
    img.onload = () => { setThumbSrc(url); setThumbStatus('ok'); };
    img.onerror = () => setThumbStatus('error');
    img.src = url;
    return () => { img.onload = null; img.onerror = null; };
  }, [item.path, serverUri, requestedSize]);

  // Build sections, collecting remaining keys for "Other"
  const shown = new Set<string>();
  const sections: { label: string; entries: [string, unknown][] }[] = [];

  for (const section of SECTION_ORDER) {
    const entries: [string, unknown][] = [];
    for (const k of section.keys) {
      const v = meta[k];
      if (v !== undefined && v !== null && String(v).trim() !== '') {
        entries.push([k, v]);
        shown.add(k);
      }
    }
    if (entries.length > 0) sections.push({ label: section.label, entries });
  }

  // Remaining keys (skip internal/noise)
  const skipPrefixes = ['thinfilm_', 'vae_embedding', 'OrganicSalt_SMILES', 'MetalSalt_SMILES'];
  const otherEntries: [string, unknown][] = Object.entries(meta)
    .filter(([k, v]) => {
      if (shown.has(k)) return false;
      if (skipPrefixes.some(p => k.startsWith(p))) return false;
      if (v === null || v === undefined || String(v).trim() === '') return false;
      if (k.startsWith('Sample ') || k.startsWith('M1 ') || k.startsWith('Slit')) return false;
      return true;
    })
    .slice(0, 40);

  if (otherEntries.length > 0) sections.push({ label: 'Other', entries: otherEntries });

  return (
    <div className="flex h-full w-full flex-col bg-slate-950 text-slate-200">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b"
        style={{ borderColor: '#334155', background: '#1e293b' }}
      >
        <div className="min-w-0">
          <p className="text-sm font-semibold truncate" style={{ color: '#e2e8f0' }}>
            {item.sample}
          </p>
          <p className="text-xs truncate mt-0.5" style={{ color: '#64748b' }}>
            {item.path}
          </p>
        </div>
        <button
          onClick={onClose}
          className="ml-2 shrink-0 p-1 rounded hover:bg-slate-700 transition-colors"
          style={{ color: '#94a3b8' }}
        >
          <X size={14} />
        </button>
      </div>

      {/* Thumbnail — scales with the panel width */}
      <div
        className="shrink-0 border-b flex items-center justify-center"
        style={{ borderColor: '#334155', background: '#0a1120', height: thumbStatus === 'error' ? 0 : previewHeight, overflow: 'hidden' }}
      >
        {thumbStatus === 'loading' && (
          <div className="flex flex-col items-center gap-2">
            <ImageIcon size={22} style={{ color: '#334155' }} />
            <span className="text-xs" style={{ color: '#475569' }}>Loading preview…</span>
          </div>
        )}
        {thumbStatus === 'ok' && thumbSrc && (
          <img
            src={thumbSrc}
            alt="Array preview"
            style={{ maxHeight: previewHeight, maxWidth: '100%', objectFit: 'contain', imageRendering: 'pixelated' }}
          />
        )}
        {thumbStatus === 'error' && null}
      </div>

      {/* Scrollable metadata */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4" style={{ minHeight: 0 }}>
        {sections.map(sec => (
          <div key={sec.label}>
            <p
              className="text-xs font-semibold uppercase tracking-wide mb-1.5"
              style={{ color: '#3b82f6' }}
            >
              {sec.label}
            </p>
            <table className="w-full text-xs border-collapse">
              <tbody>
                {sec.entries.map(([k, v]) => (
                  <tr key={k} className="align-top">
                    <td
                      className="pr-2 py-0.5 font-medium whitespace-nowrap"
                      style={{ color: '#94a3b8', maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis' }}
                    >
                      {displayKey(k)}
                    </td>
                    <td
                      className="py-0.5 break-words"
                      style={{ color: '#e2e8f0', wordBreak: 'break-word' }}
                    >
                      {formatValue(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>

      {(onOpenInAnnotate || onOpenInTrain) && (
        <div className="shrink-0 flex flex-col gap-2 px-4 py-3 border-t" style={{ borderColor: '#334155' }}>
          {onOpenInAnnotate && (
            <button
              type="button"
              onClick={onOpenInAnnotate}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium bg-sky-600 text-white hover:bg-sky-500 transition-colors"
            >
              <PencilSimple size={16} />
              Open in Annotate
            </button>
          )}
          {onOpenInTrain && (
            <button
              type="button"
              onClick={onOpenInTrain}
              title="Fine-tune or run inference on this annotated sample"
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium bg-violet-600 text-white hover:bg-violet-500 transition-colors"
            >
              <Brain size={16} />
              Open in Train
            </button>
          )}
        </div>
      )}
    </div>
  );
}
