/**
 * trainModelConfig — builds the `model` payload TrainPage sends to
 * `/api/train/start` and `/api/train/estimate-batch`, and validates it before
 * either request goes out. Both call sites used to assemble this same shape
 * (and re-check the same precondition) by hand; this is the one place it
 * happens now, so the two can't drift out of sync with each other.
 *
 * dlsia TUNet only — DINOv3 LoRA (and its checkpoint picker) is deferred, so
 * there is only one model family to build a config for.
 */
import type { HyperparamsState } from '@/components/train/HyperparamsPanel';
import { validateImageSize } from '@/lib/trainConstraints';

export interface TunetModelConfig {
  model_family: 'dlsia_tunet';
  hyperparams: {
    epochs: number;
    lr: number;
    depth: number;
    base_channels: number;
    growth_rate: number;
    batch_size: number;
    image_size: number;
    flip_augment: boolean;
    tiling: boolean;
  };
}

export type TrainModelConfig = TunetModelConfig;

/** Build the `model` field of a train/estimate-batch request. */
export function buildModelConfig(hp: HyperparamsState): TrainModelConfig {
  return {
    model_family: 'dlsia_tunet',
    hyperparams: {
      epochs: hp.epochs,
      lr: hp.lr,
      depth: hp.depth,
      base_channels: hp.base_channels,
      growth_rate: hp.growth_rate,
      batch_size: hp.batch_size,
      image_size: hp.image_size,
      flip_augment: hp.flip_augment,
      tiling: hp.tiling,
    },
  };
}

/**
 * Validate a training config before either "Start training" or "Estimate
 * batch size" fires. Returns a message to show the user, or null when the
 * config is ready to submit.
 */
export function validateTrainConfig(hp: HyperparamsState): string | null {
  return validateImageSize('dlsia_tunet', hp.image_size, hp.tiling ? 'Patch size' : 'Image size');
}

/**
 * A string that changes iff any input to the batch-size probe's memory
 * footprint changes: the image/patch size (tiling toggles whether patches are
 * cut at native resolution, which changes the tensor shapes the probe
 * actually measures).
 *
 * TrainPage stashes this when a probe starts and compares it against the
 * current config when the probe finishes — a mismatch means the user changed
 * something mid-measurement, so the result no longer describes what's about
 * to be submitted and must not be silently adopted into `batch_size`.
 */
export function trainConfigSignature(hp: Pick<HyperparamsState, 'image_size' | 'tiling'>): string {
  return JSON.stringify([hp.image_size, hp.tiling]);
}
