import { describe, expect, it } from 'vitest';
import { buildModelConfig, trainConfigSignature, validateTrainConfig } from './trainModelConfig';
import type { HyperparamsState } from '@/components/train/HyperparamsPanel';
import type { DinoCheckpoint } from '@/hooks/useTrainCapability';

const HP: HyperparamsState = {
  epochs: 30, lr: 1e-4, batch_size: 2, image_size: 512, flip_augment: true, tiling: true,
  lora_rank: 8, lora_alpha: 16,
  depth: 4, base_channels: 8, growth_rate: 1.5,
};

const CHECKPOINT: DinoCheckpoint = {
  file_name: 'dinov3_vits16.pth', arch: 'vits16', weights: 'lvd1689m', embed_dim: 384, size_bytes: 123,
};

describe('buildModelConfig', () => {
  it('emits arch/checkpoint/lora fields for dinov3_lora, and no depth/base_channels/growth_rate', () => {
    const model = buildModelConfig('dinov3_lora', CHECKPOINT, HP);
    expect(model).toEqual({
      model_family: 'dinov3_lora',
      arch: 'vits16',
      checkpoint: 'dinov3_vits16.pth',
      hyperparams: {
        epochs: 30, lr: 1e-4, lora_rank: 8, lora_alpha: 16,
        batch_size: 2, image_size: 512, flip_augment: true, tiling: true,
      },
    });
    expect(model).not.toHaveProperty('hyperparams.depth');
    expect(model).not.toHaveProperty('hyperparams.base_channels');
    expect(model).not.toHaveProperty('hyperparams.growth_rate');
  });

  it('emits depth/base_channels/growth_rate for dlsia_tunet, and no arch/checkpoint/lora', () => {
    const model = buildModelConfig('dlsia_tunet', null, HP);
    expect(model).toEqual({
      model_family: 'dlsia_tunet',
      hyperparams: {
        epochs: 30, lr: 1e-4, depth: 4, base_channels: 8, growth_rate: 1.5,
        batch_size: 2, image_size: 512, flip_augment: true, tiling: true,
      },
    });
    expect(model).not.toHaveProperty('arch');
    expect(model).not.toHaveProperty('checkpoint');
    expect(model).not.toHaveProperty('hyperparams.lora_rank');
  });

  it('carries the tiling flag through for both families', () => {
    expect(buildModelConfig('dinov3_lora', CHECKPOINT, { ...HP, tiling: false }).hyperparams.tiling).toBe(false);
    expect(buildModelConfig('dlsia_tunet', null, { ...HP, tiling: false }).hyperparams.tiling).toBe(false);
  });

  it('throws if asked to build a dinov3_lora config with no checkpoint selected', () => {
    // Guards against calling this out of order — validateTrainConfig should
    // always be checked first and should have already caught this case.
    expect(() => buildModelConfig('dinov3_lora', null, HP)).toThrow(/checkpoint/i);
  });
});

describe('validateTrainConfig', () => {
  it('accepts a well-formed dinov3_lora config', () => {
    expect(validateTrainConfig('dinov3_lora', CHECKPOINT, HP)).toBeNull();
  });

  it('accepts a well-formed dlsia_tunet config', () => {
    expect(validateTrainConfig('dlsia_tunet', null, HP)).toBeNull();
  });

  it('rejects a dinov3_lora config with no checkpoint selected', () => {
    expect(validateTrainConfig('dinov3_lora', null, HP)).toMatch(/checkpoint/i);
  });

  it('does not require a checkpoint for dlsia_tunet', () => {
    expect(validateTrainConfig('dlsia_tunet', null, { ...HP, image_size: 512 })).toBeNull();
  });

  it('rejects an out-of-range image size, naming it "Patch size" when tiling is on', () => {
    const msg = validateTrainConfig('dinov3_lora', CHECKPOINT, { ...HP, image_size: 32, tiling: true });
    expect(msg).toContain('Patch size');
  });

  it('names the field "Image size" when tiling is off', () => {
    const msg = validateTrainConfig('dinov3_lora', CHECKPOINT, { ...HP, image_size: 32, tiling: false });
    expect(msg).toContain('Image size');
  });

  it('checks the checkpoint before the image size, matching the stricter of the two prior call sites', () => {
    // Both a missing checkpoint AND an out-of-range size are wrong here; the
    // checkpoint message should win so the user fixes the more fundamental
    // problem first.
    const msg = validateTrainConfig('dinov3_lora', null, { ...HP, image_size: 32 });
    expect(msg).toMatch(/checkpoint/i);
  });
});

describe('trainConfigSignature', () => {
  it('is identical for the same config across separate calls', () => {
    expect(trainConfigSignature('dinov3_lora', CHECKPOINT, HP)).toBe(trainConfigSignature('dinov3_lora', CHECKPOINT, HP));
  });

  it('changes when the model family changes', () => {
    expect(trainConfigSignature('dinov3_lora', CHECKPOINT, HP)).not.toBe(trainConfigSignature('dlsia_tunet', null, HP));
  });

  it('changes when the checkpoint (arch or file) changes', () => {
    const other: DinoCheckpoint = { ...CHECKPOINT, file_name: 'dinov3_vitb16.pth', arch: 'vitb16' };
    expect(trainConfigSignature('dinov3_lora', CHECKPOINT, HP)).not.toBe(trainConfigSignature('dinov3_lora', other, HP));
  });

  it('changes when image_size changes', () => {
    expect(trainConfigSignature('dlsia_tunet', null, HP))
      .not.toBe(trainConfigSignature('dlsia_tunet', null, { ...HP, image_size: 256 }));
  });

  it('changes when tiling toggles', () => {
    expect(trainConfigSignature('dlsia_tunet', null, HP))
      .not.toBe(trainConfigSignature('dlsia_tunet', null, { ...HP, tiling: !HP.tiling }));
  });

  it('is unaffected by hyperparameters that do not change the probe’s memory footprint', () => {
    const withUnrelatedChanges: HyperparamsState = { ...HP, epochs: 999, lr: 1, batch_size: 32 };
    expect(trainConfigSignature('dlsia_tunet', null, HP))
      .toBe(trainConfigSignature('dlsia_tunet', null, withUnrelatedChanges));
  });
});
