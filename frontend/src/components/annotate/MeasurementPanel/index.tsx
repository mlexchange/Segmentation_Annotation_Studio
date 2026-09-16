/**
 * MeasurementPanel — quantitative readout for the current selection.
 *
 * Geometry (area / perimeter / centroid / bbox) is computed client-side from the
 * selected shapes; intensity statistics (min/max/mean/std of raw values inside
 * the region) are fetched from POST /api/measure. An optional pixel size adds
 * physical units. Shows a hint when nothing is selected.
 */
import { useEffect, useMemo, useState } from 'react';
import { Ruler } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { measureRegion } from '@/lib/measure';
import CollapsibleSection from '@/components/common/CollapsibleSection';

interface MeasurementPanelProps {
  sourceKey: string | null;
}

interface IntensityStats {
  pixel_count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
}

/** Two-column stat row. */
function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-gray-500">{label}</span>
      <span className="tabular-nums text-gray-800">{value}</span>
    </div>
  );
}

export default function MeasurementPanel({ sourceKey }: MeasurementPanelProps) {
  const { meta, currentSlice } = useDatasetStore();
  const byImage = useAnnotationStore((s) => s.byImage);
  const selectedShapeIds = useToolStore((s) => s.selectedShapeIds);
  const [pixelSize, setPixelSize] = useState('');
  const [unit, setUnit] = useState('µm');
  const [intensity, setIntensity] = useState<IntensityStats | null>(null);
  const [intensityLoading, setIntensityLoading] = useState(false);

  // Selected shapes on the current slice.
  const selectedShapes = useMemo(() => {
    if (!sourceKey) return [];
    const shapes = byImage[sourceKey]?.[currentSlice] ?? [];
    const ids = new Set(selectedShapeIds);
    return shapes.filter((s) => ids.has(s.id));
  }, [sourceKey, byImage, currentSlice, selectedShapeIds]);

  const geom = useMemo(
    () => (meta ? measureRegion(selectedShapes, meta.width, meta.height) : null),
    [selectedShapes, meta],
  );

  // Fetch raw-intensity stats for the current selection (debounced).
  useEffect(() => {
    if (!sourceKey || selectedShapes.length === 0) { setIntensity(null); return; }
    let cancelled = false;
    setIntensityLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/measure?source_key=${encodeURIComponent(sourceKey)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ slice_index: currentSlice, shapes: selectedShapes }),
        });
        if (!cancelled && res.ok) setIntensity(await res.json());
        else if (!cancelled) setIntensity(null);
      } catch {
        if (!cancelled) setIntensity(null);
      } finally {
        if (!cancelled) setIntensityLoading(false);
      }
    }, 300);
    return () => { cancelled = true; clearTimeout(t); };
  }, [sourceKey, selectedShapes, currentSlice]);

  const px = Number(pixelSize);
  const cal = pixelSize.trim() !== '' && Number.isFinite(px) && px > 0 ? px : null;
  const area = (v: number) => (cal ? `${(v * px * px).toPrecision(4)} ${unit}²` : `${v.toFixed(0)} px²`);
  const len = (v: number) => (cal ? `${(v * px).toPrecision(4)} ${unit}` : `${v.toFixed(1)} px`);

  return (
    <CollapsibleSection title="Measure" icon={<Ruler size={14} />}>
      {!geom || geom.count === 0 ? (
        <p className="text-xs text-gray-500">Select one or more regions to measure.</p>
      ) : (
        <div className="flex flex-col gap-1 text-xs">
          <Row label={`Regions`} value={String(geom.count)} />
          <Row label="Area" value={area(geom.areaPx)} />
          <Row label="Perimeter" value={geom.perimeterPx == null ? '—' : len(geom.perimeterPx)} />
          {geom.centroid && (
            <Row label="Centroid" value={`${geom.centroid.x.toFixed(0)}, ${geom.centroid.y.toFixed(0)}`} />
          )}
          {geom.bbox && (
            <Row label="Bounds" value={`${geom.bbox.w.toFixed(0)}×${geom.bbox.h.toFixed(0)} px`} />
          )}

          <div className="my-1 border-t border-gray-100" />
          <div className="text-[10px] uppercase tracking-wide text-gray-500">Intensity (raw)</div>
          {intensityLoading ? (
            <p className="text-gray-500">Measuring…</p>
          ) : intensity && intensity.pixel_count > 0 ? (
            <>
              <Row label="Mean ± SD" value={`${fmt(intensity.mean)} ± ${fmt(intensity.std)}`} />
              <Row label="Min / Max" value={`${fmt(intensity.min)} / ${fmt(intensity.max)}`} />
              <Row label="Pixels" value={intensity.pixel_count.toLocaleString()} />
            </>
          ) : (
            <p className="text-gray-500">—</p>
          )}

          {/* Pixel-size calibration (optional). */}
          <div className="mt-1.5 flex items-center gap-1">
            <span className="text-gray-500">Pixel size</span>
            <input
              type="number"
              min={0}
              step="any"
              value={pixelSize}
              onChange={(e) => setPixelSize(e.target.value)}
              placeholder="—"
              className="w-14 rounded border border-gray-200 px-1 py-0.5 text-right tabular-nums"
            />
            <input
              value={unit}
              onChange={(e) => setUnit(e.target.value)}
              className="w-10 rounded border border-gray-200 px-1 py-0.5"
            />
            <span className="text-gray-500">/px</span>
          </div>
        </div>
      )}
    </CollapsibleSection>
  );
}

/** Format an intensity value compactly (integers plain, else 3 sig figs). */
function fmt(v: number | null): string {
  if (v == null) return '—';
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(4);
}
