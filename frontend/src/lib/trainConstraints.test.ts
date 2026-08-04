import { describe, expect, it } from 'vitest';
import { IMAGE_SIZE_CONSTRAINTS, validateImageSize } from './trainConstraints';

describe('validateImageSize', () => {
  it('accepts the documented defaults for both families', () => {
    expect(validateImageSize('dinov3_lora', 512, 'Patch size')).toBeNull();
    expect(validateImageSize('dlsia_tunet', 512, 'Patch size')).toBeNull();
  });

  it('rejects a value below the minimum and names the range', () => {
    const msg = validateImageSize('dinov3_lora', 32, 'Patch size');
    expect(msg).toContain('Patch size');
    expect(msg).toContain('224');
    expect(msg).toContain('1024');
    expect(msg).toContain('32'); // echoes what was entered
  });

  it('rejects a value above the maximum', () => {
    expect(validateImageSize('dinov3_lora', 2048, 'Patch size')).toContain('between 224 and 1024');
    expect(validateImageSize('dlsia_tunet', 4096, 'Image size')).toContain('between 64 and 2048');
  });

  it('rejects a non-multiple and suggests a valid nearby value', () => {
    const msg = validateImageSize('dinov3_lora', 500, 'Patch size');
    expect(msg).toContain('multiple of 16');
    expect(msg).toContain('496'); // floor(500/16)*16
  });

  it('uses each family’s own divisor', () => {
    // 320 is a valid multiple of 16 but not of 64.
    expect(validateImageSize('dinov3_lora', 320, 'Patch size')).toBeNull();
    expect(validateImageSize('dlsia_tunet', 320, 'Patch size')).toBeNull();
    expect(validateImageSize('dlsia_tunet', 336, 'Patch size')).toContain('multiple of 64');
  });

  it('uses the label it is given, so the message matches the visible control', () => {
    expect(validateImageSize('dinov3_lora', 32, 'Image size')).toContain('Image size');
    expect(validateImageSize('dinov3_lora', 32, 'Patch size')).toContain('Patch size');
  });

  it('rejects non-integers and non-finite input', () => {
    expect(validateImageSize('dinov3_lora', 512.5, 'Patch size')).toContain('whole number');
    expect(validateImageSize('dinov3_lora', Number.NaN, 'Patch size')).toContain('whole number');
  });

  it('never suggests a value below the family minimum', () => {
    // 230 floors to 224 for DINOv3, which is exactly the minimum — not below it.
    const msg = validateImageSize('dinov3_lora', 230, 'Patch size');
    expect(msg).toContain('224');
  });

  it('exposes constraints that match the backend schema', () => {
    expect(IMAGE_SIZE_CONSTRAINTS.dinov3_lora).toMatchObject({ min: 224, max: 1024, multipleOf: 16 });
    expect(IMAGE_SIZE_CONSTRAINTS.dlsia_tunet).toMatchObject({ min: 64, max: 2048, multipleOf: 64 });
  });
});
