/**
 * LearnedDenoiserPanel — train a self-supervised denoiser (Noise2Noise or
 * Noise2Void) on the open sample's raw slices, then preview a saved run on
 * the current slice. Rendered by DenoisePanel when its mode toggle is set to
 * "Learned denoiser"; split out as a sibling because between the scheme
 * picker, the training-scope selector, the train job, and the run
 * picker/preview it would roughly triple DenoisePanel's size.
 *
 * Unlike the classical filters, neither scheme needs any annotations —
 * that's the whole appeal for a first-pass denoise. Training goes through
 * the SAME `/api/train/start` endpoint segmentation training already uses,
 * distinguished by `task: "denoising"` and `classes: []` (a teammate is
 * landing that split server-side; see the TODOs below for the exact fields
 * still to confirm once it merges).
 *
 * The training-scope selector (current slice / a range / the whole volume)
 * mirrors ApplyModelPanel's fine-tune-scope control as closely as possible,
 * including its inline validation-message behavior — see
 * `lib/denoiserTrainScope.ts`, which holds that logic as pure functions so
 * both panels' variants can't drift silently.
 */
import { useEffect, useMemo, useState } from 'react';
import { Archive, MagicWand, Student } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useDenoiseStore } from '@/stores/denoiseStore';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { useTrainRuns } from '@/hooks/useTrainRuns';
import { useExportJob } from '@/hooks/useExportJob';
import { isDenoiserRun } from '@/lib/runCompatibility';
import { buildSliceUrl } from '@/hooks/useImageSlice';
import {
  denoiserScopeBlockedReason, resolveDenoiserScopeIndices,
  type DenoiserScheme, type DenoiserTrainScope,
} from '@/lib/denoiserTrainScope';
import JobProgressBar from '@/components/train/JobProgressBar';
import type { DenoiseBakeTarget } from './DenoiseBakeModal';
import type { RenderOpts } from '@/stores/datasetStore';

// Mirrors DenoisePanel's CROP_PREVIEW_SIZE (kept local rather than imported to
// avoid a circular import between the two sibling components — same value,
// same rationale: a 1:1 centre crop for previewing a slow full-slice op).
const MODEL_PREVIEW_CROP_SIZE = 768;

// Matches TrainPage's DEFAULT_TUNET_HYPERPARAMS — a dlsia denoiser reuses the
// TUNet architecture's hyperparameter shape (see the model payload below).
const DEFAULT_DENOISER_HYPERPARAMS = {
  epochs: 60, lr: 1e-3, depth: 4, base_channels: 8, growth_rate: 1.5,
  batch_size: 4, image_size: 512, flip_augment: true, tiling: true,
};

const SCHEME_INFO: Record<DenoiserScheme, { label: string; caveat: string }> = {
  n2n: {
    label: 'Noise2Noise',
    caveat: 'Trains on pairs of adjacent slices, treating each as an independent noisy '
      + 'view of the same structure. Works well when neighboring slices really are '
      + 'similar — if they differ a lot (sparse sampling, fast-changing structure), '
      + 'it can blur away real z-structure along with the noise.',
  },
  n2v: {
    label: 'Noise2Void',
    caveat: 'Learns to predict each pixel from its surroundings without ever seeing that '
      + "pixel's own value, so it targets uncorrelated per-pixel noise. It will NOT "
      + 'remove correlated artifacts like ring or streak patterns common in CT '
      + 'reconstructions — those repeat across neighboring pixels in a way this scheme '
      + "can't distinguish from real structure.",
  },
  ae: {
    label: 'Autoencoder (bottleneck)',
    caveat: 'A plain encoder/decoder with no skip connections, trained to reconstruct each '
      + 'slice through a narrow latent bottleneck. Nothing is added to your data — the noise '
      + "is squeezed out because it's the part that can't fit through the bottleneck. Trains "
      + 'on a single slice and needs no annotations. Because it compresses, it discards some '
      + 'fine real detail too, so it usually looks blurrier than Noise2Void.',
  },
};

/**
 * Human label for a saved denoiser run: its scheme, plus the bottleneck for an
 * autoencoder run.
 *
 * `model_config` is an untyped `Record<string, unknown>` (the backend records
 * `{architecture, depth, base_channels, training_scheme, ...}`), hence the
 * guards. Two runs used to be indistinguishable in the picker unless their
 * scheme differed — with two architectures and a tunable bottleneck, that got
 * worse, so the compression is part of the label rather than hidden.
 */
export function describeDenoiserRun(run: { model_config: Record<string, unknown> }): string {
  const scheme = run.model_config.training_scheme;
  const label = typeof scheme === 'string'
    ? SCHEME_INFO[scheme as DenoiserScheme]?.label ?? scheme
    : 'Denoiser';
  const compression = run.model_config.ae_compression;
  return typeof compression === 'number' ? `${label} ${compression}x` : label;
}

export interface LearnedDenoiserPanelProps {
  source: string | null;
  kind: string | null;
  serverUri: string | null;
  currentSlice: number;
  nSlices: number;
  renderOpts: RenderOpts;
  /** Opens the "save a denoised copy" flow for the selected run (Tiled only). */
  onBake?: (target: DenoiseBakeTarget) => void;
}

export default function LearnedDenoiserPanel({
  source, kind, serverUri, currentSlice, nSlices, renderOpts, onBake,
}: LearnedDenoiserPanelProps) {
  const { capability } = useTrainCapability();
  const { runs: allRuns, invalidate: refreshRuns } = useTrainRuns();
  const denoiserRuns = useMemo(() => allRuns.filter(isDenoiserRun), [allRuns]);

  const [scheme, setScheme] = useState<DenoiserScheme>('n2v');
  // How tightly the 'ae' scheme's bottleneck compresses (input values : latent
   // values). Matches schemas.DlsiaDenoiserConfig.ae_compression's default and
  // its 4..64 bounds.
  const [compression, setCompression] = useState(16);
  const [epochs, setEpochs] = useState(DEFAULT_DENOISER_HYPERPARAMS.epochs);
  // Default 'all': the most useful default for a self-supervised denoiser —
  // more slices only helps it generalize, and (unlike segmentation) there's
  // no annotation cost to including every slice.
  const [trainScope, setTrainScope] = useState<DenoiserTrainScope>('all');
  const [trainRangeStart, setTrainRangeStart] = useState(0);
  const [trainRangeEnd, setTrainRangeEnd] = useState(Math.max(0, nSlices - 1));
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    setTrainRangeEnd(Math.max(0, nSlices - 1));
  }, [nSlices]);

  // Scoped per source, like ApplyModelPanel's fine-tune job: navigating away
  // and back should only ever reattach to a job for the sample that's open now.
  const { state: trainJob, startJob: startTrainJob, reset: resetTrainJob } = useExportJob(
    source ? `annotate:denoise-train:${source}` : undefined,
  );

  const scopeIndices = resolveDenoiserScopeIndices(trainScope, currentSlice, trainRangeStart, trainRangeEnd, nSlices);
  const scopeBlockedReason = denoiserScopeBlockedReason(scopeIndices, scheme);

  useEffect(() => {
    if (trainJob.status === 'done') refreshRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainJob.status]);

  const handleTrain = () => {
    setLocalError(null);
    if (!source || !kind) return;
    if (scopeBlockedReason) {
      setLocalError(scopeBlockedReason);
      return;
    }
    resetTrainJob();
    void startTrainJob('/api/train/start', {
      sources: [{
        kind: kind === 'tiled' ? 'tiled' : 'local',
        source,
        server_uri: serverUri,
        // No annotations exist or are needed — Noise2Noise/Noise2Void train
        // directly on these slices' raw pixels. `slices` is reused here only
        // to name WHICH raw slice indices are in scope (empty shape lists),
        // since ExportSourceItem has no other field for "just these indices."
        // TODO: confirm against the backend once denoiser training lands —
        // a dedicated `slice_indices` field may replace this.
        slices: Object.fromEntries(scopeIndices.map((i) => [String(i), []])),
        split_by_slice: {},
        negative_slices: [],
      }],
      classes: [],
      // Matches schemas.TrainRequest.task (Literal["segmentation","denoising"],
      // defaulting to "segmentation" so existing requests are unaffected).
      task: 'denoising',
      model: {
        // `model_family`, NOT `kind` — ModelConfig discriminates on
        // `model_family` (schemas.py's ModelConfig union). Sending `kind` here
        // fails validation with "Unable to extract tag using discriminator
        // 'model_family'". Note the *render* union nearby DOES use `kind`,
        // which is what made this an easy mistake to make.
        model_family: 'dlsia_denoiser',
        hyperparams: { ...DEFAULT_DENOISER_HYPERPARAMS, epochs },
        training_scheme: scheme,
        // The bottleneck autoencoder is a different NETWORK, not just a
        // different objective — the backend pairs the two and rejects any other
        // combination (schemas.DlsiaDenoiserConfig), since pure reconstruction
        // on a skip-connected model would just learn to copy its input.
        ...(scheme === 'ae'
          ? { architecture: 'cnn_ae', ae_compression: compression }
          : {}),
      },
    });
  };

  const handleCancelTrain = () => {
    if (trainJob.jobId) void fetch(`${API_BASE}/api/train/cancel/${trainJob.jobId}`, { method: 'POST' });
  };

  // Same reasoning as TrainPage's `deviceBusy`/`tunetNeedsDlsia`: a dlsia
  // denoiser needs both torch and dlsia, and can't overlap with another job
  // holding the server's ML_LOCK.
  const trainRunning = trainJob.status === 'running';
  const deviceBusy = trainRunning || capability.busy;
  const denoiserNeedsDlsia = !capability.dlsia.available;
  const trainDisabled = !source || deviceBusy || !capability.torch_available || denoiserNeedsDlsia || !!scopeBlockedReason;

  // --- Run picker + apply-for-preview -------------------------------------
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [appliedRunId, setAppliedRunId] = useState<string | null>(null);
  const [showPreviewCrop, setShowPreviewCrop] = useState(false);

  // Default to (and self-heal onto) the newest denoiser run, mirroring
  // ApplyModelPanel's run-selection effect.
  useEffect(() => {
    if (denoiserRuns.length === 0) {
      if (selectedRunId !== null) setSelectedRunId(null);
      return;
    }
    if (selectedRunId === null || !denoiserRuns.some((r) => r.run_id === selectedRunId)) {
      setSelectedRunId(denoiserRuns[0].run_id);
    }
  }, [denoiserRuns, selectedRunId]);

  // Adopt a freshly-trained run once training finishes.
  useEffect(() => {
    if (trainJob.status !== 'done') return;
    const newRunId = typeof trainJob.result?.run_id === 'string' ? trainJob.result.run_id : null;
    if (newRunId) setSelectedRunId(newRunId);
  }, [trainJob.status, trainJob.result]);

  // A different run or sample invalidates whatever was last applied for preview.
  useEffect(() => {
    setAppliedRunId(null);
  }, [selectedRunId, source]);

  // Applying to the MAIN CANVAS rather than only to the inset below is what
  // makes a denoised slice actually inspectable: the inset shows the server PNG
  // raw, while the canvas puts brightness/contrast/levels/gamma on top via the
  // GPU filter chain. On low-contrast data (this sample's slices span only
  // ~0-107 of the 8-bit range) the untouched PNG reads as near-black, so the
  // inset alone was effectively unviewable.
  const denoise = useDenoiseStore((s) => s.denoise);
  const setDenoise = useDenoiseStore((s) => s.setDenoise);
  const resetDenoise = useDenoiseStore((s) => s.resetDenoise);
  const canvasRunId = denoise.method === 'model' ? denoise.runId ?? null : null;
  const showingOnCanvas = canvasRunId !== null && canvasRunId === selectedRunId;

  const handleApplyToCanvas = () => {
    if (!selectedRunId) return;
    // Never pass `crop` here: the canvas maps shapes onto the sample's full
    // dimensions, so a cropped image would render every annotation offset.
    setDenoise({ method: 'model', strength: 0, runId: selectedRunId });
  };

  const handleApplyForPreview = () => {
    if (selectedRunId) setAppliedRunId(selectedRunId);
  };

  const previewUrl = useMemo(() => {
    if (!source || !kind || !appliedRunId) return null;
    return buildSliceUrl(source, kind, currentSlice, renderOpts, serverUri, {
      method: 'model',
      strength: 0, // unused for a learned denoiser — DenoiseOpts requires the field
      runId: appliedRunId,
      ...(showPreviewCrop ? { crop: MODEL_PREVIEW_CROP_SIZE } : {}),
    });
  }, [source, kind, currentSlice, renderOpts, serverUri, appliedRunId, showPreviewCrop]);

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Scheme</span>
        {(Object.keys(SCHEME_INFO) as DenoiserScheme[]).map((value) => (
          <label
            key={value}
            className={`flex items-start gap-2 px-2.5 py-2 rounded-md border cursor-pointer transition-colors ${
              scheme === value ? 'border-sky-400 bg-sky-50' : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <input
              type="radio" name="denoiser-scheme" className="mt-0.5 shrink-0 accent-sky-600"
              checked={scheme === value} onChange={() => setScheme(value)}
            />
            <div className="min-w-0">
              <p className="text-xs font-medium text-gray-700">{SCHEME_INFO[value].label}</p>
              <p className="mt-0.5 text-[11px] leading-snug text-gray-500">{SCHEME_INFO[value].caveat}</p>
            </div>
          </label>
        ))}
      </div>

      {scheme === 'ae' && (
        <div className="flex flex-col gap-0.5 pt-0.5">
          <label className="flex items-center gap-2 text-xs text-gray-600">
            <span className="shrink-0">Compression</span>
            <input
              // Powers of two across the backend's 4..64 range: each step
              // halves the bottleneck, which is a visible change. A linear
              // slider would spend most of its travel on differences too small
              // to see.
              type="range" min={2} max={6} step={1}
              value={Math.round(Math.log2(compression))}
              disabled={trainRunning}
              onChange={(e) => setCompression(2 ** Number(e.target.value))}
              className="flex-1 accent-violet-600 disabled:opacity-50"
            />
            <span className="w-9 shrink-0 text-right tabular-nums">{compression}x</span>
          </label>
          <p className="pl-1 text-[11px] leading-snug text-gray-400">
            Tighter removes more noise but blurs more detail.
          </p>
        </div>
      )}

      <div className="flex flex-col gap-1 pt-0.5">
        <span className="text-xs text-gray-400">Train on</span>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {([
            ['current', `This slice (${currentSlice})`],
            ['range', 'Slice range'],
            ['all', `All slices (${nSlices})`],
          ] as const).map(([value, label]) => (
            <label key={value} className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
              <input
                type="radio" name="denoiser-train-scope" className="accent-sky-600"
                checked={trainScope === value} disabled={trainRunning}
                onChange={() => setTrainScope(value)}
              />
              {label}
            </label>
          ))}
        </div>
        {trainScope === 'range' && (
          <div className="flex items-center gap-2 pl-5 text-xs text-gray-600">
            <input
              type="number" min={0} max={nSlices - 1} value={trainRangeStart} disabled={trainRunning}
              onChange={(e) => setTrainRangeStart(Number(e.target.value))}
              className="w-16 rounded border border-gray-300 bg-white px-1.5 py-1 disabled:opacity-50"
            />
            <span>to</span>
            <input
              type="number" min={0} max={nSlices - 1} value={trainRangeEnd} disabled={trainRunning}
              onChange={(e) => setTrainRangeEnd(Number(e.target.value))}
              className="w-16 rounded border border-gray-300 bg-white px-1.5 py-1 disabled:opacity-50"
            />
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={handleTrain}
          disabled={trainDisabled}
          title={
            !capability.torch_available
              ? 'Training is unavailable: torch is not installed on this server'
              : denoiserNeedsDlsia
                ? 'Training a dlsia denoiser needs dlsia installed on this server'
                : capability.busy && !trainRunning
                  ? 'Another training or inference job is using the device'
                  : (scopeBlockedReason ?? 'Train a self-supervised denoiser on the slices above')
          }
          className="flex flex-1 items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-violet-600 text-white hover:bg-violet-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Student size={16} />
          {trainRunning ? 'Training…' : 'Train'}
        </button>
        <label className="flex items-center gap-1 text-xs text-gray-500" title="Training epochs">
          <input
            type="number" min={1} max={1000} value={epochs} disabled={trainRunning}
            onChange={(e) => setEpochs(Math.max(1, Math.min(1000, Number(e.target.value) || 1)))}
            className="w-14 rounded-md border border-gray-300 bg-white px-1.5 py-1 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
          />
          ep
        </label>
        {trainRunning && (
          <button
            type="button" onClick={handleCancelTrain}
            className="shrink-0 px-3 py-2 text-xs rounded-md border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {denoiserNeedsDlsia && (
        <p className="text-[11px] text-amber-700">
          dlsia isn't installed on this server. Install it (<code className="font-mono">uv pip install dlsia</code>)
          to train a learned denoiser.
        </p>
      )}
      {scopeBlockedReason && !trainRunning && (
        <p className="text-[11px] text-amber-700">{scopeBlockedReason}</p>
      )}
      {localError && <p className="text-xs text-red-600 break-words">{localError}</p>}

      <JobProgressBar job={trainJob} unit="epochs" />
      {trainJob.status === 'done' && (
        <p className="text-xs text-emerald-600">
          Saved run <span className="font-mono">{String(trainJob.result?.run_id ?? '')}</span>.
        </p>
      )}

      <hr />

      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Preview a trained run</span>
        {denoiserRuns.length === 0 ? (
          <p className="text-xs text-gray-400">No denoiser runs yet — train one above.</p>
        ) : (
          <>
            <select
              value={selectedRunId ?? ''}
              onChange={(e) => setSelectedRunId(e.target.value || null)}
              className="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500"
            >
              {denoiserRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {describeDenoiserRun(run)}
                  {' — '}{new Date(run.created_at).toLocaleString()}
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={showingOnCanvas ? resetDenoise : handleApplyToCanvas}
              disabled={!selectedRunId}
              title={
                showingOnCanvas
                  ? 'Go back to the original slice on the canvas'
                  : 'Run this denoiser on the current slice and show it on the main canvas, '
                    + 'with your brightness/contrast settings applied'
              }
              className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                showingOnCanvas
                  ? 'bg-amber-100 text-amber-800 hover:bg-amber-200'
                  : 'bg-sky-600 text-white hover:bg-sky-500'
              }`}
            >
              <MagicWand size={16} />
              {showingOnCanvas ? 'Show original' : 'Show on canvas'}
            </button>

            {showingOnCanvas && (
              <p className="text-[11px] leading-snug text-amber-700">
                The canvas is showing denoised pixels. Display only — annotations and exports still
                use the original data, but the magic wand and SAM do see this image.
              </p>
            )}

            <button
              type="button"
              onClick={handleApplyForPreview}
              disabled={!selectedRunId}
              title={
                'Render a small crop here instead — faster than a full-slice pass, but shown '
                + 'without your brightness/contrast adjustments'
              }
              className="flex items-center justify-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Quick crop check
            </button>

            {appliedRunId && (
              <label className="flex items-start gap-1.5 text-[11px] text-gray-600 cursor-pointer">
                <input
                  type="checkbox" checked={showPreviewCrop}
                  onChange={(e) => setShowPreviewCrop(e.target.checked)}
                  className="mt-0.5 accent-sky-600"
                />
                <span>
                  Fast 1:1 preview
                  <span className="text-gray-400">
                    {' '}— runs the model on a {MODEL_PREVIEW_CROP_SIZE}px centre crop instead of the whole
                    slice; a full-slice forward pass can be slow.
                  </span>
                </span>
              </label>
            )}

            {previewUrl && (
              <div className="rounded-md border border-gray-200 overflow-hidden bg-black">
                <img src={previewUrl} alt="Learned denoiser preview" className="block w-full" />
              </div>
            )}

            {appliedRunId && (
              <p className="text-[11px] leading-snug text-amber-700">
                Display only, and shown here rather than on the canvas — it doesn't change
                annotations, the canvas, or the saved data.
              </p>
            )}

            {onBake && (
              <button
                type="button"
                onClick={() => {
                  const run = denoiserRuns.find((r) => r.run_id === selectedRunId);
                  onBake({
                    method: 'model',
                    label: run ? describeDenoiserRun(run) : 'Trained denoiser',
                    strength: 0, // unused for a model bake
                    runId: selectedRunId ?? undefined,
                  });
                }}
                disabled={!selectedRunId}
                title="Run this denoiser over every slice and save the result as a new dataset"
                className="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-sky-100 text-sky-700 hover:bg-sky-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Archive size={16} />
                Save denoised copy…
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
