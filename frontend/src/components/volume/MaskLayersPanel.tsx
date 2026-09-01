/**
 * MaskLayersPanel — sidebar overlay for the 3-D view's two mask/annotation
 * layers: "Fast (iPred)" and "Deep (dlsia)". The viewer's own `loadMask`/
 * `loadMaskFromArray`/etc. are slot-neutral (0 | 1, no idea what produced a
 * mask — see `WebGpuViewerInstance`'s own doc comment); this panel is where
 * that meaning ("fast" vs "deep", which Tiled container each comes from)
 * actually lives, kept out of the vendored viewer entirely.
 *
 * Both slots are Tiled-backed for now (`buildMaskZarrUrl`), pointed at the
 * two independent `<source>__masks` / `<source>__masks_deep` containers
 * `tiled_mask_sync.write_masks_to_tiled` writes (see its `container_suffix`
 * doc). A zero-latency, no-Tiled-round-trip live preview for the Fast slot
 * (rasterizing current Annotate-tab predictions client-side via
 * `loadMaskFromArray`) is a deliberately deferred follow-up — this repo has
 * no existing client-side volume-rasterization code to build on, unlike the
 * fork's `labelVolume.ts`/`volumeDims.ts`, which assume a server-side
 * downsampled-raw-volume endpoint this app's `/volume` page does not have
 * (it streams the real Tiled Zarr pyramid directly instead).
 */
import { useEffect, useState } from 'react';
import { Eye, EyeSlash, CircleNotch } from '@phosphor-icons/react';
import { buildMaskZarrUrl } from '@/lib/zarrUrl';
import type { WebGpuViewerInstance } from './VolumeViewer';

type MaskClasses = NonNullable<ReturnType<WebGpuViewerInstance['getMaskClasses']>>;

interface SlotConfig {
  slot: 0 | 1;
  label: string;
  suffix: '' | '_deep';
}

const SLOTS: SlotConfig[] = [
  { slot: 0, label: 'Fast (iPred)', suffix: '' },
  { slot: 1, label: 'Deep (dlsia)', suffix: '_deep' },
];

interface SlotState {
  loading: boolean;
  error: string | null;
  loaded: boolean;
  enabled: boolean;
  opacity: number;
  classes: MaskClasses;
}

const initialSlotState: SlotState = {
  loading: false,
  error: null,
  loaded: false,
  enabled: true,
  opacity: 0.6,
  classes: [],
};

interface MaskLayersPanelProps {
  instance: WebGpuViewerInstance | null;
  kind: string | null;
  source: string | null;
  serverUri: string | null;
}

export default function MaskLayersPanel({ instance, kind, source, serverUri }: MaskLayersPanelProps) {
  const [state, setState] = useState<Record<0 | 1, SlotState>>({
    0: initialSlotState,
    1: initialSlotState,
  });

  // A new dataset invalidates every previously-loaded mask — the viewer
  // itself remounts on source change (see VolumePage's `key={url}`), so
  // there is nothing to explicitly unload here, only local status to reset.
  useEffect(() => {
    setState({ 0: initialSlotState, 1: initialSlotState });
  }, [source, serverUri]);

  if (!instance) return null;

  const load = async (cfg: SlotConfig) => {
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
        [cfg.slot]: { ...s[cfg.slot], loading: false, loaded: true, classes, error: classes.length === 0 ? 'Loaded, but no classes found (every voxel is background, or this dataset has no mask pushed yet).' : null },
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
