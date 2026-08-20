/**
 * VolumePage — 3D volume view of the currently-open sample: a raycast render
 * of the raw intensity stack with a colored, per-class-selectable overlay of
 * the LIVE annotations (not saved masks — whatever is drawn in Annotate right
 * now). The raw volume comes from `/api/image/volume` (downsampled
 * server-side); the label volume is rasterized client-side from
 * `annotationStore` onto the EXACT same voxel grid (see `volumeDimsFor`), so
 * the two must never be computed independently — a mismatch here would
 * silently render the overlay offset from the data it describes.
 */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import { ArrowsClockwise, Cube } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { buildSourceKey } from '@/lib/sourceKey';
import { useVolume } from '@/hooks/useVolume';
import {
  QUALITY_LADDER, effectiveMaxDim, volumeDimsFor, type VolumeDims,
} from '@/lib/volume/volumeDims';
import { buildLabelVolume } from '@/lib/volume/labelVolume';
import type { VolumeMode } from '@/lib/volume/volumeRenderer';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import VolumeCanvas from '@/components/volume/VolumeCanvas';
import ClassVisibilityList from '@/components/volume/ClassVisibilityList';

const DEFAULT_MAX_DIM = 256;
/** Debounces label-volume rebuilds after an annotation edit — a single
 *  keystroke-adjacent stroke can touch several slices in quick succession, and
 *  rebuilding on every one would repeatedly rasterize the same result. */
const LABEL_REBUILD_DEBOUNCE_MS = 250;

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return [255, 255, 255];
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

export default function VolumePage() {
  const navigate = useNavigate();
  const { kind, source, serverUri, meta, renderOpts } = useDatasetStore();
  const classes = useClassStore((s) => s.classes);

  const sourceKey = source && kind ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri) : null;
  // Reference-stable per edit (the store rebuilds this object only when the
  // slice map for THIS source actually changes) — a good effect dependency.
  const slicesForSource = useAnnotationStore((s) => (sourceKey ? s.byImage[sourceKey] : undefined));

  const [maxDim, setMaxDim] = useState<number>(DEFAULT_MAX_DIM);
  const [mode, setMode] = useState<VolumeMode>('both');
  const [rawOpacity, setRawOpacity] = useState(1);
  const [labelOpacity, setLabelOpacity] = useState(0.6);
  const [windowLo, setWindowLo] = useState(0);
  const [windowHi, setWindowHi] = useState(1);
  const [zScale, setZScale] = useState(1);
  const [shading, setShading] = useState(false);
  const [shadingStrength, setShadingStrength] = useState(0.8);
  const [resetNonce, setResetNonce] = useState(0);
  const [visible, setVisible] = useState<Record<number, boolean>>({});

  // Reconcile visibility map when classes are added/removed — default a new
  // class to visible rather than silently dropping it from the LUT.
  useEffect(() => {
    setVisible((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const cls of classes) {
        if (!(cls.classId in next)) {
          next[cls.classId] = true;
          changed = true;
        }
      }
      for (const key of Object.keys(next)) {
        if (!classes.some((c) => c.classId === Number(key))) {
          delete next[Number(key)];
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [classes]);

  const volumeRenderOpts = useMemo(
    () => ({ norm: renderOpts.norm, scale: renderOpts.scale, vminPct: renderOpts.vminPct, vmaxPct: renderOpts.vmaxPct }),
    [renderOpts.norm, renderOpts.scale, renderOpts.vminPct, renderOpts.vmaxPct]
  );
  const volume = useVolume(source, kind, serverUri, maxDim, volumeRenderOpts);

  // classOrder is intentionally classStore's own order — buildLabelVolume
  // paints later classes over earlier ones on overlap, matching how classes
  // higher in the list already draw on top in Annotate's canvas.
  const classOrder = useMemo(() => classes.map((c) => c.classId), [classes]);

  const [labelVolume, setLabelVolume] = useState<{ data: Uint8Array; dims: VolumeDims } | null>(null);

  useEffect(() => {
    if (!volume.data) {
      setLabelVolume(null);
      return;
    }
    const dims = volume.data.dims;
    const timer = window.setTimeout(() => {
      setLabelVolume({ data: buildLabelVolume(slicesForSource, classOrder, dims), dims });
    }, LABEL_REBUILD_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [slicesForSource, classOrder, volume.data]);

  // 256x1 RGBA LUT: index 0 = background (fully transparent), index i+1 =
  // classOrder[i]'s display color, alpha 0 when that class is hidden.
  const lut = useMemo(() => {
    const table = new Uint8Array(1024);
    classes.forEach((cls, i) => {
      const idx = Math.min(i + 1, 255);
      const [r, g, b] = hexToRgb(cls.color);
      const isVisible = visible[cls.classId] ?? true;
      table[idx * 4] = r;
      table[idx * 4 + 1] = g;
      table[idx * 4 + 2] = b;
      table[idx * 4 + 3] = isVisible ? 255 : 0;
    });
    return table;
  }, [classes, visible]);

  const canvasOptions = useMemo(
    () => ({
      mode,
      rawOpacity,
      labelOpacity,
      windowLo,
      windowHi: Math.max(windowHi, windowLo + 0.01),
      zScale,
      shading,
      shadingStrength,
    }),
    [mode, rawOpacity, labelOpacity, windowLo, windowHi, zScale, shading, shadingStrength]
  );

  if (!meta) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-sky-200">
        <p>No sample loaded. Pick a sample to view in 3D.</p>
        <button
          type="button"
          onClick={() => navigate('/browse')}
          className="px-4 py-2 rounded-md bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 transition-colors"
        >
          Go to Browse
        </button>
      </div>
    );
  }

  // The EFFECTIVE quality, not the requested one: the server clamps to the
  // voxel budget with this same ladder, so the label volume rasterized here has
  // to be built on the clamped grid too. Using the raw request would put the
  // overlay on a finer grid than the raw data whenever a capped quality is
  // chosen — an overlay silently offset from the data it describes.
  const effectiveDim = effectiveMaxDim(meta.nSlices, meta.height, meta.width, maxDim);
  const expectedDims = volumeDimsFor(meta.nSlices, meta.height, meta.width, effectiveDim);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <div className="w-72 flex-shrink-0 border-r border-gray-200 bg-white overflow-y-auto overflow-x-hidden p-3 flex flex-col gap-4">
        <div className="flex items-center gap-2 text-gray-800">
          <Cube size={20} className="text-sky-600" />
          <h1 className="text-sm font-semibold">3D Volume</h1>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Mode</span>
          <div className="flex rounded-md border border-gray-200 overflow-hidden">
            {([['both', 'Both'], ['raw', 'Raw'], ['labels', 'Labels']] as const).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setMode(value)}
                className={`flex-1 px-2 py-1.5 text-xs font-medium transition-colors ${
                  mode === value ? 'bg-sky-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <ClassVisibilityList classes={classes} visible={visible} onChange={setVisible} />

        <div className="flex flex-col gap-3 border-t border-gray-100 pt-3">
          <DebouncedSlider
            label="Raw opacity"
            value={rawOpacity}
            min={0}
            max={1}
            step={0.01}
            onChange={setRawOpacity}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <DebouncedSlider
            label="Label opacity"
            value={labelOpacity}
            min={0}
            max={1}
            step={0.01}
            onChange={setLabelOpacity}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <DebouncedSlider
            label="Window min"
            value={windowLo}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => setWindowLo(Math.min(v, windowHi - 0.01))}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <DebouncedSlider
            label="Window max"
            value={windowHi}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => setWindowHi(Math.max(v, windowLo + 0.01))}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <DebouncedSlider
            label="Z scale"
            value={zScale}
            min={0.2}
            max={5}
            step={0.05}
            onChange={setZScale}
            format={(v) => `${v.toFixed(2)}x`}
            allowLog
          />
          <label
            className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none"
            title="Shades the volume using the raw intensity gradient as an approximate surface normal, so grain boundaries catch light instead of rendering as a flat, evenly-lit gel. Costs more to render."
          >
            <input
              type="checkbox"
              checked={shading}
              onChange={(e) => setShading(e.target.checked)}
              className="h-3.5 w-3.5 cursor-pointer accent-sky-600"
            />
            Realistic lighting
          </label>
          {shading && (
            <DebouncedSlider
              label="Light strength"
              value={shadingStrength}
              min={0}
              max={1}
              step={0.01}
              onChange={setShadingStrength}
              format={(v) => `${Math.round(v * 100)}%`}
            />
          )}
        </div>

        <div className="flex flex-col gap-1.5 border-t border-gray-100 pt-3">
          <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Quality</span>
          <select
            value={maxDim}
            onChange={(e) => setMaxDim(Number(e.target.value))}
            className="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500"
          >
            {QUALITY_LADDER.map((q) => {
              // Show the grid this quality actually yields for THIS volume. A
              // thin stack reaches the top of the ladder cheaply, while a cubic
              // one gets clamped to the voxel budget — without this the same
              // "1024px" label would mean wildly different things per dataset,
              // and a clamped choice would look like it did nothing.
              const eff = effectiveMaxDim(meta.nSlices, meta.height, meta.width, q);
              const d = volumeDimsFor(meta.nSlices, meta.height, meta.width, eff);
              const mb = (d.nz * d.ny * d.nx) / (1024 * 1024);
              return (
                <option key={q} value={q}>
                  {q}px{q === DEFAULT_MAX_DIM ? ' (default)' : ''}
                  {` — ${d.nx}x${d.ny}x${d.nz}, ${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`}
                  {eff !== q ? ' (capped)' : ''}
                </option>
              );
            })}
          </select>
          {effectiveDim !== maxDim && (
            <p className="text-[11px] leading-snug text-amber-700">
              Capped to {effectiveDim}px to stay within the volume memory budget — this stack is
              too cubic for the full grid.
            </p>
          )}
        </div>

        <button
          type="button"
          onClick={() => setResetNonce((n) => n + 1)}
          className="flex items-center justify-center gap-2 px-3 py-1.5 rounded-md border border-gray-200 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
        >
          <ArrowsClockwise size={14} />
          Reset view
        </button>

        <div className="mt-auto border-t border-gray-100 pt-3 text-xs text-gray-400 space-y-1">
          <p>
            {expectedDims.nx}×{expectedDims.ny}×{expectedDims.nz} voxels · {meta.dtype}
          </p>
          {meta.isRgb && <p>RGB shown as luminance.</p>}
          {volume.data && volume.data.skippedZ.length > 0 && (
            <p className="text-amber-600">
              {volume.data.skippedZ.length} slice{volume.data.skippedZ.length === 1 ? '' : 's'} unreadable — shown empty.
            </p>
          )}
        </div>
      </div>

      {/* Viewport */}
      <div className="relative flex-1 bg-slate-900">
        {/* Keying on resetNonce remounts the canvas (fresh renderer + default
            orbit state) — simplest way to reset the view without threading an
            imperative ref through VolumeCanvas for this one rarely-used action. */}
        <VolumeCanvas
          key={resetNonce}
          raw={volume.data ? { data: volume.data.data, dims: volume.data.dims } : null}
          label={labelVolume}
          lut={lut}
          options={canvasOptions}
        />

        {volume.isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/70 text-slate-200 text-sm">
            <div className="flex flex-col items-center gap-2 rounded-lg bg-slate-800/60 px-4 py-3">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-500 border-t-sky-400" />
              <p>Loading volume — large stacks may take a while.</p>
            </div>
          </div>
        )}

        {volume.isError && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 text-slate-200 text-sm p-4">
            <div className="flex flex-col items-center gap-3 rounded-lg bg-slate-800/70 px-4 py-3 max-w-md text-center">
              <p className="break-words">
                {volume.error instanceof Error ? volume.error.message : 'Failed to load the volume.'}
              </p>
              <button
                type="button"
                onClick={() => volume.refetch()}
                className="px-3 py-1.5 rounded-md bg-sky-600 text-white text-xs font-medium hover:bg-sky-700 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

