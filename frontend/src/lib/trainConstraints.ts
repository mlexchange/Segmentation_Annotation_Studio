/**
 * trainConstraints — the bounds the backend enforces on training hyperparameters,
 * mirrored client-side so the Train tab can show the valid range and reject a bad
 * value before it becomes an opaque 422.
 *
 * Keep in sync with `DinoHyperParams` / `TunetHyperParams` in backend/schemas.py —
 * that remains the authority; this only exists to fail earlier and more legibly.
 */

export type TrainFamily = 'dinov3_lora' | 'dlsia_tunet';

export interface SizeConstraint {
  min: number;
  max: number;
  /** The side must be divisible by this. */
  multipleOf: number;
  /** Why the divisor exists, for the error message. */
  reason: string;
}

export const IMAGE_SIZE_CONSTRAINTS: Record<TrainFamily, SizeConstraint> = {
  // A ViT sees the input as a grid of 16px tokens; below ~224 there are too few
  // tokens for the backbone to have any spatial context to work with.
  dinov3_lora: { min: 224, max: 1024, multipleOf: 16, reason: "DINOv3's 16px patch grid" },
  // TUNet halves the size once per depth level; 64 covers the deepest allowed net.
  dlsia_tunet: { min: 64, max: 2048, multipleOf: 64, reason: "TUNet's downsampling steps" },
};

/**
 * Validate the image/patch size for `family`, returning a message to show the
 * user, or null when the value is acceptable.
 *
 * `label` names the field as the UI currently labels it ("Patch size" when
 * tiling, "Image size" otherwise) so the message matches what's on screen.
 */
export function validateImageSize(family: TrainFamily, value: number, label: string): string | null {
  const c = IMAGE_SIZE_CONSTRAINTS[family];
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    return `${label} must be a whole number of pixels.`;
  }
  if (value < c.min || value > c.max) {
    return `${label} must be between ${c.min} and ${c.max} px for this model (you entered ${value}).`;
  }
  if (value % c.multipleOf !== 0) {
    const lower = Math.floor(value / c.multipleOf) * c.multipleOf;
    const nearest = Math.max(c.min, lower < c.min ? c.min : lower);
    return `${label} must be a multiple of ${c.multipleOf} (${c.reason}) — try ${nearest}.`;
  }
  return null;
}
