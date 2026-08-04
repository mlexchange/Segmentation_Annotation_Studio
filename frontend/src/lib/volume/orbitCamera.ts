/**
 * Orbit ("arcball-lite") camera math for the hand-written WebGL2 volume
 * raycaster. Pure math only — no WebGL/canvas code, no external matrix
 * library (three.js/gl-matrix are NOT installed and NOT required).
 *
 * Coordinate system: right-handed, Y-up, matching standard OpenGL/WebGL
 * convention.
 *
 * Spherical -> cartesian convention (see `orbitEye`):
 *   x = target.x + distance * sin(phi) * sin(theta)
 *   y = target.y + distance * cos(phi)
 *   z = target.z + distance * sin(phi) * cos(theta)
 * `theta` is the azimuth (rotation around +Y, measured from +Z toward +X),
 * `phi` is the polar angle measured DOWN from +Y (phi=0 is the north pole
 * "top", phi=PI/2 is the equator/horizon, phi=PI is the south pole
 * "bottom"). `phi` is kept strictly inside (EPS, PI - EPS) so the camera
 * never sits at a pole, which would make the up vector degenerate
 * (gimbal lock) in `lookAt`.
 *
 * Matrix layout: all mat4 values here are **column-major**, the GL-native
 * convention (matches `gl.uniformMatrix4fv`/GLSL). A Float32Array of length
 * 16 stores column 0 in indices [0..3], column 1 in [4..7], etc., i.e.
 * `m[col * 4 + row]`. Because this already matches GL's native layout, a
 * consumer should call `gl.uniformMatrix4fv(location, false, m)` — transpose
 * MUST be `false`. `multiply4(a, b)` computes `a * b` under this layout,
 * meaning `multiply4(a, b)` applied to a column vector `v` is `a * (b * v)`
 * (b is applied first) — the usual "transform composition" order used by
 * view/projection matrix stacks.
 */

export interface OrbitState {
  theta: number; // azimuth, radians, unbounded (wraps freely)
  phi: number; // polar angle from +Y axis, radians, kept in (EPS, PI - EPS)
  distance: number; // camera distance from target, > 0
  target: [number, number, number];
}

export const DEFAULT_ORBIT_STATE: OrbitState = {
  theta: 0,
  phi: Math.PI / 2,
  distance: 3,
  target: [0, 0, 0],
};

/** Smallest/largest allowed polar angle — keeps the camera off the poles. */
const PHI_EPS = 0.01;

/** Absolute distance clamp bounds (world units), documented per spec. */
const MIN_DISTANCE = 0.5;
const MAX_DISTANCE = 50;

/**
 * Pointer-drag-to-radians sensitivity. Tuned so a full-width drag on an
 * ~800px canvas is roughly one full theta revolution:
 * 2*PI / 800 ~= 0.0078. We use a slightly rounder 0.005 rad/px (a full
 * revolution takes ~1257px, a bit more than one canvas-width drag, which
 * feels less twitchy for fine orbit control).
 */
const ROTATE_SENSITIVITY = 0.005;

/**
 * Zoom rate constant for the exponential dolly: `distance *= exp(ZOOM_K *
 * wheelDelta)`. Chosen so a typical wheel "notch" (deltaY ~= 100) changes
 * distance by roughly 5%.
 */
const ZOOM_K = 0.0005;

/**
 * Pan speed constant: pixel-to-world-unit conversion is
 * `distance * PAN_SPEED` per pixel, so panning feels consistent regardless
 * of current zoom level.
 */
const PAN_SPEED = 0.002;

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Camera eye position in world space for the given orbit state. See the
 *  top-of-file comment for the exact spherical->cartesian convention used. */
export function orbitEye(state: OrbitState): [number, number, number] {
  const { theta, phi, distance, target } = state;
  const sinPhi = Math.sin(phi);
  return [
    target[0] + distance * sinPhi * Math.sin(theta),
    target[1] + distance * Math.cos(phi),
    target[2] + distance * sinPhi * Math.cos(theta),
  ];
}

/** Rotate the orbit by raw pointer-move deltas (pixels). Pure function. */
export function rotate(state: OrbitState, dxPixels: number, dyPixels: number): OrbitState {
  const theta = state.theta + dxPixels * ROTATE_SENSITIVITY;
  const phi = clamp(state.phi + dyPixels * ROTATE_SENSITIVITY, PHI_EPS, Math.PI - PHI_EPS);
  return { ...state, theta, phi };
}

/** Dolly zoom by a raw wheel deltaY. Positive delta (scroll down) zooms out. */
export function zoom(state: OrbitState, wheelDelta: number): OrbitState {
  const distance = clamp(state.distance * Math.exp(ZOOM_K * wheelDelta), MIN_DISTANCE, MAX_DISTANCE);
  return { ...state, distance };
}

/**
 * Pan the target in the camera's local right/up plane. Only `target`
 * changes; theta/phi/distance are passed through untouched.
 */
export function pan(state: OrbitState, dxPixels: number, dyPixels: number): OrbitState {
  const { theta, phi, distance, target } = state;
  const panScale = distance * PAN_SPEED;

  // Camera forward direction (eye -> target), derived from the same
  // spherical convention as orbitEye, negated (eye = target + dir * distance
  // => forward = -dir).
  const sinPhi = Math.sin(phi);
  const dir: [number, number, number] = [sinPhi * Math.sin(theta), Math.cos(phi), sinPhi * Math.cos(theta)];

  // World up, used to derive an orthonormal right/up basis for the camera.
  const worldUp: [number, number, number] = [0, 1, 0];

  // right = normalize(cross(dir, worldUp)) -- dir is the eye->target axis,
  // so this matches the right vector used by lookAt() for the same eye.
  let right = cross(dir, worldUp);
  right = normalize(right);

  // camUp = cross(right, dir) -- orthonormal "up" in the camera's local
  // right/up plane.
  const camUp = normalize(cross(right, dir));

  // Dragging right (+dx) should move the target so the scene appears to
  // follow the cursor, i.e. pan the target in the -right direction; dragging
  // down (+dy) pans target in the +up direction for the same reason (drag
  // down => world appears to move down => target moves up relative to eye).
  const newTarget: [number, number, number] = [
    target[0] - right[0] * dxPixels * panScale + camUp[0] * dyPixels * panScale,
    target[1] - right[1] * dxPixels * panScale + camUp[1] * dyPixels * panScale,
    target[2] - right[2] * dxPixels * panScale + camUp[2] * dyPixels * panScale,
  ];

  return { ...state, target: newTarget };
}

// ---------------------------------------------------------------------------
// Vec3 helpers (internal)
// ---------------------------------------------------------------------------

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-8) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function subtract(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ---------------------------------------------------------------------------
// mat4 helpers — column-major Float32Array(16), GL-native layout.
// Index convention: m[col * 4 + row].
// ---------------------------------------------------------------------------

/** Standard right-handed perspective projection (column-major, GL clip
 *  space z in [-1, 1], matching the classic OpenGL projection matrix). */
export function perspective(fovYRadians: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovYRadians / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

/** Standard right-handed lookAt view matrix (column-major). */
export function lookAt(
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number]
): Float32Array {
  const zAxis = normalize(subtract(eye, target)); // forward (camera looks down -zAxis)
  let xAxis = normalize(cross(up, zAxis)); // right
  if (xAxis[0] === 0 && xAxis[1] === 0 && xAxis[2] === 0) {
    // up is parallel to zAxis (degenerate) -- fall back to a stable axis.
    xAxis = normalize(cross([0, 0, 1], zAxis));
  }
  const yAxis = cross(zAxis, xAxis); // recomputed up, orthonormal

  const out = new Float32Array(16);
  out[0] = xAxis[0];
  out[1] = yAxis[0];
  out[2] = zAxis[0];
  out[3] = 0;

  out[4] = xAxis[1];
  out[5] = yAxis[1];
  out[6] = zAxis[1];
  out[7] = 0;

  out[8] = xAxis[2];
  out[9] = yAxis[2];
  out[10] = zAxis[2];
  out[11] = 0;

  out[12] = -dot(xAxis, eye);
  out[13] = -dot(yAxis, eye);
  out[14] = -dot(zAxis, eye);
  out[15] = 1;
  return out;
}

/** Column-major 4x4 matrix multiply: returns a * b (b applied first to a
 *  column vector, i.e. result * v == a * (b * v)). */
export function multiply4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

/**
 * Invert a column-major 4x4 matrix via the standard cofactor/adjugate
 * method. If the matrix is singular (determinant ~= 0), returns the 4x4
 * identity matrix rather than throwing or dividing by zero — callers that
 * need to detect this should check the determinant themselves; for this
 * module's use (inverting well-formed view-projection matrices) singularity
 * should not occur in practice.
 */
export function invert4(m: Float32Array): Float32Array {
  const a = m;
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

  if (Math.abs(det) < 1e-12) {
    // Singular: return identity rather than dividing by ~0.
    const identity = new Float32Array(16);
    identity[0] = identity[5] = identity[10] = identity[15] = 1;
    return identity;
  }
  det = 1 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/** Default vertical field of view: ~50 degrees, in radians. */
const DEFAULT_FOVY = (50 * Math.PI) / 180;
const DEFAULT_NEAR = 0.01;
const DEFAULT_FAR = 1000;

/**
 * Convenience: full view-projection matrix + its inverse + the eye position,
 * for a raycaster that needs to unproject screen pixels back into world
 * rays. `up` is always [0,1,0] (safe because `rotate` keeps phi off the
 * poles, so the eye is never collinear with world up).
 */
export function viewProj(
  state: OrbitState,
  aspect: number,
  fovYRadians: number = DEFAULT_FOVY,
  near: number = DEFAULT_NEAR,
  far: number = DEFAULT_FAR
): { vp: Float32Array; invVp: Float32Array; eye: [number, number, number] } {
  const eye = orbitEye(state);
  const view = lookAt(eye, state.target, [0, 1, 0]);
  const proj = perspective(fovYRadians, aspect, near, far);
  const vp = multiply4(proj, view);
  const invVp = invert4(vp);
  return { vp, invVp, eye };
}
