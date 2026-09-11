/**
 * FeatureChannelsPanel — the Assist-stage sidebar panel.
 *
 * Compositions (feature-bank recipes) are surfaced as named presets so most
 * users never need the underlying node graph. The full graph editor
 * (`CompositionPanel`) is still there for anyone who wants a custom recipe —
 * it just lives behind a collapsed "Advanced" disclosure instead of its own
 * tab, since composing a feature graph is a power-user path, not the default.
 *
 * Presets whose modules aren't ready (e.g. TomoJEPA checkpoints not present —
 * see ipred/models/README.md) are greyed out rather than left to fail at
 * Compute time; only SlimSAM ships auto-vendored today, so it's the
 * recommended default.
 */
import { useEffect, useMemo, useState } from 'react';
import { CaretLeft, CaretRight, CircleNotch, Stack, Warning } from '@phosphor-icons/react';
import type { FeatureJobInfo } from '@/hooks/useFeatureChannels';
import { useIpredStore } from '@/stores/ipredStore';
import { listIpredCompositions, listIpredModules, type CompositionDoc, type FeatureModuleInfo } from '@/lib/ipredApi';
import CompositionPanel from '@/components/CompositionPanel';
import CollapsibleSection from '@/components/common/CollapsibleSection';
import { cn } from '@/lib/utils';

/** Plain-language labels for the 7 built-in compositions, by id. */
const PRESET_LABELS: Record<string, string> = {
  'comp-skimage': 'Fast (skimage only)',
  'comp-skimage-slimsam': 'Texture-aware (+ SlimSAM)',
  'comp-slimsam-clahe': 'Texture-aware, contrast-enhanced',
  'comp-skimage-mark25': 'Deep features (TomoJEPA Mark25)',
  'comp-mark25-clahe': 'Deep features, contrast-enhanced (Mark25)',
  'comp-skimage-mark11': 'Deep features (TomoJEPA Mark11)',
  'comp-mark11-clahe': 'Deep features, contrast-enhanced (Mark11)',
};

/** One-line explanation of what each preset trades off, shown under the picker. */
const PRESET_HINTS: Record<string, string> = {
  'comp-skimage': 'Multiscale edge/texture filters only. Cheapest, no model download.',
  'comp-skimage-slimsam': 'Skimage filters + SlimSAM vision-encoder embeddings (PCA-reduced). Recommended default.',
  'comp-slimsam-clahe': 'SlimSAM embeddings on a contrast-equalized slice — helps on low-contrast data.',
  'comp-skimage-mark25': 'Skimage filters + TomoJEPA Mark25 embeddings. Requires a private checkpoint.',
  'comp-mark25-clahe': 'TomoJEPA Mark25 on a contrast-equalized slice. Requires a private checkpoint.',
  'comp-skimage-mark11': 'Skimage filters + TomoJEPA Mark11 embeddings. Requires a private checkpoint.',
  'comp-mark11-clahe': 'TomoJEPA Mark11 on a contrast-equalized slice. Requires a private checkpoint.',
};

const RECOMMENDED_PRESET_ID = 'comp-skimage-slimsam';

export interface FeatureChannelsPanelProps {
  job: FeatureJobInfo | null;
  channelIndex: number | null;
  computing: boolean;
  error: string | null;
  onCompute: () => void;
  onSelectChannel: (index: number | null) => void;
  onCycle: (delta: number) => void;
  onOriginal: () => void;
  disabled?: boolean;
}

export default function FeatureChannelsPanel({
  job,
  channelIndex,
  computing,
  error,
  onCompute,
  onSelectChannel,
  onCycle,
  onOriginal,
  disabled = false,
}: FeatureChannelsPanelProps) {
  const preferredCompositionId = useIpredStore((s) => s.preferredCompositionId);
  const setPreferredCompositionId = useIpredStore((s) => s.setPreferredCompositionId);
  const [presets, setPresets] = useState<CompositionDoc[]>([]);
  const [modules, setModules] = useState<FeatureModuleInfo[]>([]);
  const [loadingPresets, setLoadingPresets] = useState(true);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([listIpredCompositions(), listIpredModules()])
      .then(([comps, mods]) => {
        if (cancelled) return;
        setPresets(comps);
        setModules(mods);
      })
      .catch((e) => {
        if (!cancelled) setPresetsError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoadingPresets(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const moduleById = useMemo(() => new Map(modules.map((m) => [m.id, m])), [modules]);

  /** A preset is ready only if every module it wires in is ready (e.g. weights present). */
  const presetReady = useMemo(() => {
    const out = new Map<string, boolean>();
    for (const p of presets) {
      out.set(p.id, p.nodes.every((n) => moduleById.get(n.module)?.ready !== false));
    }
    return out;
  }, [presets, moduleById]);

  const selected = presets.find((p) => p.id === preferredCompositionId) ?? null;
  const selectedReady = selected ? (presetReady.get(selected.id) ?? true) : true;
  const missingModule = selected?.nodes.find((n) => moduleById.get(n.module)?.ready === false);
  const missingModuleName = missingModule ? (moduleById.get(missingModule.module)?.name ?? missingModule.module) : null;

  // If the preferred preset isn't ready once modules load (e.g. a stale choice from
  // a previous session), fall back to the recommended one instead of a dead-end.
  useEffect(() => {
    if (loadingPresets || presets.length === 0) return;
    if (presetReady.get(preferredCompositionId) === false) {
      const fallback = presetReady.get(RECOMMENDED_PRESET_ID) ? RECOMMENDED_PRESET_ID : presets.find((p) => presetReady.get(p.id))?.id;
      if (fallback && fallback !== preferredCompositionId) setPreferredCompositionId(fallback);
    }
    // Only re-run when the readiness picture itself changes, not on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadingPresets, presetReady]);

  const n = job?.channels.length ?? 0;
  const activeLabel =
    channelIndex !== null && job ? (job.channels[channelIndex]?.label ?? `Channel ${channelIndex}`) : null;

  return (
    <CollapsibleSection
      title="Features"
      headerRight={
        n > 0 ? (
          <span className="text-[10px] text-gray-400">
            {n} ch{job?.hasSam ? ' +SAM' : ''}
            {job?.cacheHit ? ' · cache' : ''}
          </span>
        ) : undefined
      }
    >
      <label className="flex flex-col gap-0.5 text-xs text-gray-600">
        Recipe
        {loadingPresets ? (
          <div className="h-8 animate-pulse rounded border border-gray-200 bg-gray-100" />
        ) : (
          <select
            className="rounded border border-gray-200 px-2 py-1.5 text-sm text-gray-800"
            value={preferredCompositionId}
            onChange={(e) => setPreferredCompositionId(e.target.value)}
            disabled={computing || disabled}
          >
            {presets.map((c) => {
              const ready = presetReady.get(c.id) ?? true;
              return (
                <option key={c.id} value={c.id} disabled={!ready}>
                  {c.id === RECOMMENDED_PRESET_ID ? '★ ' : ''}
                  {PRESET_LABELS[c.id] ?? c.name}
                  {!ready ? ' (unavailable)' : ''}
                </option>
              );
            })}
          </select>
        )}
      </label>
      {selected && (
        <p className="text-[10px] text-gray-500 leading-snug">{PRESET_HINTS[selected.id] ?? ''}</p>
      )}
      {presetsError && <p className="text-[10px] text-red-600 leading-snug">{presetsError}</p>}

      {selected && !selectedReady && (
        <div className="flex items-start gap-1.5 rounded-md border border-amber-200 bg-amber-50 p-1.5 text-[10px] text-amber-800 leading-snug">
          <Warning size={13} className="mt-0.5 flex-shrink-0" />
          <span>
            {missingModuleName ?? 'A required model'} isn&apos;t installed yet — see{' '}
            <span className="font-mono">ipred/models/README.md</span>. Pick another recipe or install it first.
          </span>
        </div>
      )}

      <button
        type="button"
        disabled={computing || disabled || !selectedReady}
        onClick={onCompute}
        title={!selectedReady ? 'This recipe needs a model that is not installed' : undefined}
        className={cn(
          'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
          computing || disabled || !selectedReady
            ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
            : 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300',
        )}
      >
        {computing ? <CircleNotch size={14} className="animate-spin" /> : <Stack size={14} />}
        {computing ? 'Computing…' : 'Compute'}
      </button>

      {error && <p className="text-[10px] text-red-600 leading-snug break-words">{error}</p>}

      {job && n > 0 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="Previous channel"
              title="Previous channel"
              disabled={channelIndex === null}
              onClick={() => onCycle(-1)}
              className="p-1 rounded border border-gray-200 hover:bg-sky-50 disabled:opacity-40"
            >
              <CaretLeft size={14} />
            </button>
            <select
              className="flex-1 min-w-0 rounded border border-gray-200 px-1 py-1 text-xs text-gray-800"
              value={channelIndex === null ? '' : String(channelIndex)}
              onChange={(e) => {
                const v = e.target.value;
                onSelectChannel(v === '' ? null : Number(v));
              }}
            >
              <option value="">Original</option>
              {job.channels.map((ch) => (
                <option key={ch.index} value={ch.index}>
                  {ch.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-label="Next channel"
              title="Next channel"
              disabled={channelIndex === null}
              onClick={() => onCycle(1)}
              className="p-1 rounded border border-gray-200 hover:bg-sky-50 disabled:opacity-40"
            >
              <CaretRight size={14} />
            </button>
          </div>
          {activeLabel && (
            <p className="text-[10px] text-gray-500 truncate" title={activeLabel}>
              {activeLabel}
            </p>
          )}
          <button type="button" onClick={onOriginal} className="text-[11px] text-sky-700 hover:underline self-start">
            Show original
          </button>
        </div>
      )}

      <div className="border-t border-gray-100 pt-1.5">
        <button
          type="button"
          className="text-[10px] text-gray-500 hover:text-sky-700"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? '▾' : '▸'} Advanced: edit feature recipe
        </button>
        {showAdvanced && (
          <div className="mt-2">
            <CompositionPanel />
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}
