/**
 * trainModelConfig — builds the `model` payload TrainPage sends to
 * `/api/train/start` and `/api/train/estimate-batch`, and validates it before
 * either request goes out. Both call sites used to assemble this same shape
 * (and re-check the same two preconditions) by hand; this is the one place
 * it happens now, so the two can't drift out of sync with each other.
 */
import type { ModelFamily } from '@/components/train/ModelPickerPanel';
import type { HyperparamsState } from '@/components/train/HyperparamsPanel';
import type { DinoCheckpoint } from '@/hooks/useTrainCapability';
import { validateImageSize } from '@/lib/trainConstraints';

export interface DinoModelConfig {
  model_family: 'dinov3_lora';
  arch: string;
  checkpoint: string;
  hyperparams: {
    epochs: number;
    lr: number;
    lora_rank: number;
    lora_alpha: number;
    batch_size: number;
    image_size: number;
    flip_augment: boolean;
    tiling: boolean;
  };
}

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

export type TrainModelConfig = DinoModelConfig | TunetModelConfig;

/**
 * Build the `model` field of a train/estimate-batch request.
 *
 * `checkpoint` must be non-null for `dinov3_lora` — call `validateTrainConfig`
 * first and bail out on a non-null message rather than relying on this to
 * catch it; it throws here only as a last-resort guard against calling it
 * out of order.
 */
export function buildModelConfig(
  family: ModelFamily,
  checkpoint: DinoCheckpoint | null,
  hp: HyperparamsState,
): TrainModelConfig {
  if (family === 'dinov3_lora') {
    if (!checkpoint) throw new Error('buildModelConfig: no DINOv3 checkpoint selected');
    return {
      model_family: 'dinov3_lora',
      arch: checkpoint.arch,
      checkpoint: checkpoint.file_name,
      hyperparams: {
        epochs: hp.epochs,
        lr: hp.lr,
        lora_rank: hp.lora_rank,
        lora_alpha: hp.lora_alpha,
        batch_size: hp.batch_size,
        image_size: hp.image_size,
        flip_augment: hp.flip_augment,
        tiling: hp.tiling,
      },
    };
  }
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
 * batch size" fires — the fixed check order both call sites used to
 * duplicate (and, between the two, actually ran in different orders): a
 * DINOv3 run needs a selected checkpoint, then the image/patch size must be
 * in-range for the family. Returns a message to show the user, or null when
 * the config is ready to submit.
 */
export function validateTrainConfig(
  family: ModelFamily,
  checkpoint: DinoCheckpoint | null,
  hp: HyperparamsState,
): string | null {
  if (family === 'dinov3_lora' && !checkpoint) {
    return 'Select a DINOv3 checkpoint first.';
  }
  return validateImageSize(family, hp.image_size, hp.tiling ? 'Patch size' : 'Image size');
}

/**
 * A string that changes iff any input to the batch-size probe's memory
 * footprint changes: model family, DINOv3 arch/checkpoint, and the
 * image/patch size (tiling toggles whether patches are cut at native
 * resolution, which changes the tensor shapes the probe actually measures).
 *
 * TrainPage stashes this when a probe starts and compares it against the
 * current config when the probe finishes — a mismatch means the user changed
 * something mid-measurement, so the result no longer describes what's about
 * to be submitted and must not be silently adopted into `batch_size`.
 */
export function trainConfigSignature(
  family: ModelFamily,
  checkpoint: DinoCheckpoint | null,
  hp: Pick<HyperparamsState, 'image_size' | 'tiling'>,
): string {
  return JSON.stringify([family, checkpoint?.arch ?? null, checkpoint?.file_name ?? null, hp.image_size, hp.tiling]);
}
