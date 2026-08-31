/**
 * trainDenoiseOption — decides what (if anything) the "Train on denoised
 * input" checkbox contributes to a `/api/train/start` request.
 *
 * Both places that start a segmentation job (TrainPage's "Start training" and
 * ApplyModelPanel's "Fine-tune & apply") offer this checkbox and must agree
 * exactly on the answer, so the decision lives here as a pure function rather
 * than inline in either component — same reasoning as `denoiserTrainScope`.
 *
 * The setting itself comes from the Annotate tab's shared denoise store, so the
 * user trains on the very filter they were looking at. Two properties matter:
 *
 *   - OFF must be bit-identical to the behaviour that predates this option:
 *     no `denoise` key in the request body at all. `TrainRequest.denoise`
 *     defaults to None, so an absent key and an explicit null mean the same
 *     thing server-side, but omitting it keeps un-denoised requests literally
 *     unchanged.
 *   - ON must send ONLY `{method, strength}`. `DenoiseOpts` also carries `crop`
 *     (a preview-only centre crop) and `runId`; `DenoiseTrainOpts` is a strict
 *     model, so forwarding the whole object would be rejected outright — and
 *     `crop` would be nonsense for training even if it weren't.
 *
 * `'none'` and `'model'` are both rejected: there is nothing to apply for the
 * former, and the backend explicitly does not support a learned denoiser as a
 * preprocessor for another model (it would need its own run and a GPU pass per
 * slice). Guarding here as well as in the UI means a stale checked box can
 * never smuggle an invalid method into a request.
 */

/** The `denoise` field of a `/api/train/start` body (schemas.DenoiseTrainOpts). */
export interface TrainDenoiseOpts {
  method: string;
  strength: number;
}

/** Just the parts of `hooks/useImageSlice`'s `DenoiseOpts` that matter here. */
interface DenoiseSettings {
  method: string;
  strength: number;
}

/** Method + human label, as served by `capability.denoise.methods`. */
interface MethodLabel {
  method: string;
  label: string;
}

/**
 * Why the current denoise setting can't be baked into a training run, phrased
 * for display next to the checkbox — or null when it can.
 */
export function trainDenoiseBlockedReason(method: string): string | null {
  if (method === 'none') {
    return 'Pick a denoise filter in the Annotate tab first — there is nothing to apply yet.';
  }
  if (method === 'model') {
    return 'A learned denoiser cannot be used as a preprocessor for another model. Pick a classical filter in the Annotate tab.';
  }
  return null;
}

/**
 * The `denoise` fragment to spread into a `/api/train/start` body: `{}` when
 * the option is off or the setting is unusable, `{ denoise: {method, strength} }`
 * when it applies.
 *
 * Returning a spreadable fragment rather than a nullable value keeps the two
 * call sites from each re-deriving "and what do I do when it's off?" — the
 * whole point being that OFF adds no key whatsoever.
 */
export function trainDenoisePayload(
  enabled: boolean,
  denoise: DenoiseSettings,
): { denoise: TrainDenoiseOpts } | Record<string, never> {
  if (!enabled) return {};
  if (trainDenoiseBlockedReason(denoise.method) !== null) return {};
  return { denoise: { method: denoise.method, strength: denoise.strength } };
}

/**
 * Which filter the checkbox would actually bake in, e.g. `"Total variation,
 * 60%"` — so "Train on denoised input" is never ambiguous about *what*.
 * Null when the setting is unusable (the blocked reason is shown instead).
 *
 * `methods` is `capability.denoise.methods`; an unknown method falls back to
 * its raw name rather than rendering blank, since a frontend newer than the
 * server (or vice versa) shouldn't produce a nameless setting.
 */
export function trainDenoiseSummary(
  denoise: DenoiseSettings,
  methods: MethodLabel[],
): string | null {
  if (trainDenoiseBlockedReason(denoise.method) !== null) return null;
  return `${denoiseMethodLabel(denoise.method, methods)}, ${Math.round(denoise.strength * 100)}%`;
}

/** Human label for a denoise method id, falling back to the id itself. */
export function denoiseMethodLabel(method: string, methods: MethodLabel[]): string {
  return methods.find((m) => m.method === method)?.label ?? method;
}
