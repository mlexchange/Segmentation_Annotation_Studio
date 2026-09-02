/**
 * MaskLayersPanel — sidebar overlay for the 3-D view's two mask/annotation
 * layers: "Fast (iPred)" and "Deep (dlsia)". The viewer's own `loadMask`/
 * `loadMaskFromArray`/etc. are slot-neutral (0 | 1, no idea what produced a
 * mask — see `WebGpuViewerInstance`'s own doc comment); this panel is where
 * that meaning ("fast" vs "deep", which Tiled container each comes from)
 * actually lives, kept out of the vendored viewer entirely.
 *
 * Deep is always Tiled-backed (`buildMaskZarrUrl`), pointed at the
 * `<source>__masks_deep` container `tiled_mask_sync.write_masks_to_tiled`
 * writes (see its `container_suffix` doc) — there is no local equivalent for
 * a from-scratch-trained model. Fast defaults to a zero-latency, no-network
 * "Live" mode instead: `buildLiveMaskVolume` rasterizes the sample's CURRENT
 * shapes (every slice, client-side) into a coarse class-id array and loads
 * it via `loadMaskFromArray` — no "Sync masks to Tiled" step required first.
 * Fast can still be pointed at the Tiled-backed `<source>__masks` container
 * (the precise result, built by the real backend rasterizer) via its own
 * toggle.
 */
import { useEffect, useRef, useState } from 'react';
import { Eye, EyeSlash, CircleNotch } from '@phosphor-icons/react';
import { buildMaskZarrUrl } from '@/lib/zarrUrl';
import { buildLiveMaskVolume } from '@/lib/volumeMaskPreview';
import type { Shape } from '@/stores/annotationStore';
import type { WebGpuViewerInstance } from './VolumeViewer';

type MaskClasses = NonNullable<ReturnType<WebGpuViewerInstance['getMaskClasses']>>;

interface SlotConfig {
  slot: 0 | 1;
  label: string;
  suffix: '' | '_deep';
  /** Only the Fast slot has a no-Tiled-round-trip option — Deep is always a
   * from-scratch-trained model's saved output, nothing to rasterize live. */
  canLive: boolean;
}

const SLOTS: SlotConfig[] = [
  { slot: 0, label: 'Fast (iPred)', suffix: '', canLive: true },
  { slot: 1, label: 'Deep (dlsia)', suffix: '_deep', canLive: false },
];

type SourceMode = 'live' | 'tiled';

interface SlotState {
  loading: boolean;
  error: string | null;
  loaded: boolean;
  enabled: boolean;
  opacity: number;
  classes: MaskClasses;
  /** Ignored for slots where `canLive` is false. */
  mode: SourceMode;
}

const initialSlotState: SlotState = {
  loading: false,
  error: null,
  loaded: false,
  enabled: true,
  opacity: 0.6,
  classes: [],
  mode: 'live',
};

interface MaskLayersPanelProps {
  instance: WebGpuViewerInstance | null;
  kind: string | null;
  source: string | null;
  serverUri: string | null;
  /** Slot to load automatically once the viewer is ready — the "View in 3D"
   * hand-off from Train/Annotate arrives here already knowing which result
   * the user just produced, so it shouldn't need a second manual click. */
  autoLoadSlot?: 0 | 1;
  /** Current sample's shapes (every slice) for the Fast slot's "Live" mode —
   * `byImage[sourceKey]` from `annotationStore`, undefined if none open. */
  liveShapes?: Record<string, Shape[]>;
  imageWidth?: number;
  imageHeight?: number;
  nSlices?: number;
}

export default function MaskLayersPanel({
  instance, kind, source, serverUri, autoLoadSlot, liveShapes, imageWidth, imageHeight, nSlices,
}: MaskLayersPanelProps) {
  const [state, setState] = useState<Record<0 | 1, SlotState>>({
    0: initialSlotState,
    1: initialSlotState,
  });
  // Guards the auto-load effect against StrictMode's double-invoke and
  // against re-firing on every re-render once it's already kicked off once
  // for this instance.
  const autoLoadedFor = useRef<WebGpuViewerInstance | null>(null);

  // A new dataset invalidates every previously-loaded mask — the viewer
  // itself remounts on source change (see VolumePage's `key={url}`), so
  // there is nothing to explicitly unload here, only local status to reset.
  useEffect(() => {
    setState({ 0: initialSlotState, 1: initialSlotState });
    autoLoadedFor.current = null;
  }, [source, serverUri]);

  useEffect(() => {
    if (!instance || autoLoadSlot === undefined || autoLoadedFor.current === instance) return;
    autoLoadedFor.current = instance;
    const cfg = SLOTS.find((s) => s.slot === autoLoadSlot);
    if (!cfg) return;
    // A "View in 3D" hand-off means the user just pushed fresh data to
    // Tiled (or is coming from a saved dlsia run) — always the Tiled-backed
    // result here, regardless of whichever mode the Fast slot's toggle was
    // last left on.
    setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], mode: 'tiled' } }));
    void load(cfg, 'tiled');
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `load` is redefined every render but stable in effect: it only reads current instance/kind/source/serverUri via closure, matching the effect's own deps.
  }, [instance, autoLoadSlot]);

  if (!instance) return null;

  const load = async (cfg: SlotConfig, modeOverride?: SourceMode) => {
    const mode = modeOverride ?? state[cfg.slot].mode;

    if (cfg.canLive && mode === 'live') {
      const volume = imageWidth && imageHeight && nSlices
        ? buildLiveMaskVolume(liveShapes ?? {}, imageWidth, imageHeight, nSlices)
        : null;
      if (!volume) {
        setState((s) => ({
          ...s,
          [cfg.slot]: { ...s[cfg.slot], error: 'Nothing annotated for this sample yet.', loading: false },
        }));
        return;
      }
      setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], loading: true, error: null } }));
      try {
        instance.loadMaskFromArray(cfg.slot, volume.data, volume.dims);
        const classes = await waitForClasses(instance, cfg.slot);
        setState((s) => ({
          ...s,
          [cfg.slot]: { ...s[cfg.slot], loading: false, loaded: true, classes, error: null },
        }));
      } catch (err) {
        setState((s) => ({
          ...s,
          [cfg.slot]: { ...s[cfg.slot], loading: false, error: err instanceof Error ? err.message : String(err) },
        }));
      }
      return;
    }

    const { url, reason } = buildMaskZarrUrl(kind, source, serverUri, cfg.suffix);
    if (!url) {
      setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], error: reason ?? 'No source open', loading: false } }));
      return;
    }
    setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], loading: true, error: null } }));
    try {
      instance.loadMask(cfg.slot, url);
      // loadMask is fire-and-forget on the instance (see the upstream brief:
      // internal state, not a returned promise) — poll briefly for classes to
      // appear rather than assuming synchronous completion.
      const classes = await waitForClasses(instance, cfg.slot);
      setState((s) => ({
        ...s,
        [cfg.slot]: { ...s[cfg.slot], loading: false, loaded: true, classes, error: classes.length === 0 ? 'Loaded, but no classes found (every voxel is background, or nothing has been pushed to Tiled for this dataset yet).' : null },
      }));
    } catch (err) {
      setState((s) => ({
        ...s,
        [cfg.slot]: { ...s[cfg.slot], loading: false, error: err instanceof Error ? err.message : String(err) },
      }));
    }
  };

  const remove = (cfg: SlotConfig) => {
    instance.removeMask(cfg.slot);
    setState((s) => ({ ...s, [cfg.slot]: initialSlotState }));
  };

  const setOpacity = (cfg: SlotConfig, opacity: number) => {
    setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], opacity } }));
    for (const cls of state[cfg.slot].classes) {
      instance.setMaskClassOpacity(cfg.slot, cls.id, opacity);
    }
  };

  const setMode = (cfg: SlotConfig, mode: SourceMode) => {
    setState((s) => ({ ...s, [cfg.slot]: { ...s[cfg.slot], mode, error: null } }));
  };

  const toggleClass = (cfg: SlotConfig, classId: number) => {
    instance.toggleMaskClassVisible(cfg.slot, classId);
    setState((s) => ({
      ...s,
      [cfg.slot]: {
        ...s[cfg.slot],
        classes: s[cfg.slot].classes.map((c) => (c.id === classId ? { ...c, visible: !c.visible } : c)),
      },
    }));
  };

  return (
    <div className="pointer-events-none absolute left-3 top-3 z-10 flex w-72 flex-col gap-3">
      {SLOTS.map((cfg) => {
        const s = state[cfg.slot];
        return (
          <div
            key={cfg.slot}
            className="pointer-events-auto rounded-lg border border-sky-800/60 bg-sky-950/80 p-3 text-sky-200 backdrop-blur-sm"
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{cfg.label}</span>
              {s.loaded ? (
                <button
                  type="button"
                  onClick={() => remove(cfg)}
                  className="rounded px-2 py-0.5 text-xs text-sky-300 hover:bg-sky-900/60"
                >
                  Remove
                </button>
              ) : (
                <button
                  type="button"
                  disabled={s.loading}
                  onClick={() => void load(cfg)}
                  className="flex items-center gap-1 rounded bg-sky-600 px-2 py-0.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50"
                >
                  {s.loading && <CircleNotch size={12} className="animate-spin" />}
                  {s.loading ? 'Loading…' : 'Load'}
                </button>
              )}
            </div>

            {cfg.canLive && !s.loaded && (
              <div className="mb-2 flex overflow-hidden rounded border border-sky-800/60 text-[10px]">
                <button
                  type="button"
                  onClick={() => setMode(cfg, 'live')}
                  title="Rasterize this sample's current shapes directly in the browser — no Tiled sync needed, updates instantly"
                  className={`flex-1 py-1 ${s.mode === 'live' ? 'bg-sky-600 text-white' : 'bg-transparent text-sky-300 hover:bg-sky-900/60'}`}
                >
                  Live
                </button>
                <button
                  type="button"
                  onClick={() => setMode(cfg, 'tiled')}
                  title="Load the precise result from Tiled — requires Export → Sync masks to Tiled (or the iPred panel's Push to Tiled button) first"
                  className={`flex-1 py-1 ${s.mode === 'tiled' ? 'bg-sky-600 text-white' : 'bg-transparent text-sky-300 hover:bg-sky-900/60'}`}
                >
                  Tiled
                </button>
              </div>
            )}

            {s.error && <p className="mb-2 text-xs text-amber-300">{s.error}</p>}

            {s.loaded && s.classes.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <label className="flex items-center gap-2 text-xs">
                  <span className="w-14 shrink-0 opacity-70">Opacity</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={s.opacity}
                    onChange={(e) => setOpacity(cfg, Number(e.target.value))}
                    className="flex-1 accent-sky-500"
                  />
                </label>
                <div className="flex flex-col gap-0.5">
                  {s.classes.map((cls) => (
                    <button
                      key={cls.id}
                      type="button"
                      onClick={() => toggleClass(cfg, cls.id)}
                      className="flex items-center gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-sky-900/60"
                    >
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{
                          backgroundColor: `rgb(${cls.color.map((c) => Math.round(c * 255)).join(',')})`,
                          opacity: cls.visible ? 1 : 0.3,
                        }}
                      />
                      <span className={cls.visible ? '' : 'opacity-40'}>
                        Class {cls.id} <span className="opacity-60">({cls.voxelCount.toLocaleString()} vox)</span>
                      </span>
                      {cls.visible ? <Eye size={11} className="ml-auto opacity-60" /> : <EyeSlash size={11} className="ml-auto opacity-40" />}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * `loadMask` mutates viewer-internal state asynchronously with no returned
 * promise (see the upstream brief's Step 3 — it's fire-and-forget on the
 * public interface, matching how the HUD's own click handler calls it).
 * Poll `getMaskClasses` briefly rather than assuming it's ready on the next
 * tick — mirrors how `InferencePanel`/`useExportJob` poll a job status
 * elsewhere in this app rather than trusting synchronous completion.
 */
async function waitForClasses(
  instance: WebGpuViewerInstance,
  slot: 0 | 1,
  { attempts = 50, intervalMs = 100 }: { attempts?: number; intervalMs?: number } = {},
): Promise<MaskClasses> {
  for (let i = 0; i < attempts; i++) {
    const classes = instance.getMaskClasses(slot);
    if (classes !== undefined) return classes;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error('Timed out waiting for the mask to load — check the browser console for a renderer-side error.');
}
