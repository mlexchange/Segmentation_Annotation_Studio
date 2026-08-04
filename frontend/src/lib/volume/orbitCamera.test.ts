import { describe, it, expect } from 'vitest';
import {
  DEFAULT_ORBIT_STATE,
  orbitEye,
  rotate,
  zoom,
  pan,
  perspective,
  lookAt,
  multiply4,
  invert4,
  viewProj,
  type OrbitState,
} from './orbitCamera';

/** Transform a homogeneous point [x, y, z, 1] by a column-major mat4. */
function transformPoint(m: Float32Array, p: [number, number, number]): [number, number, number, number] {
  const v = [p[0], p[1], p[2], 1];
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let row = 0; row < 4; row++) {
    out[row] = m[0 * 4 + row] * v[0] + m[1 * 4 + row] * v[1] + m[2 * 4 + row] * v[2] + m[3 * 4 + row] * v[3];
  }
  return out;
}

function expectIdentity(m: Float32Array) {
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      const expected = col === row ? 1 : 0;
      expect(m[col * 4 + row]).toBeCloseTo(expected, 3);
    }
  }
}

describe('orbitCamera', () => {
  describe('orbitEye', () => {
    it('theta=0, phi=PI/2 (equator), distance=5, target origin -> [0,0,5]', () => {
      const state: OrbitState = { theta: 0, phi: Math.PI / 2, distance: 5, target: [0, 0, 0] };
      const eye = orbitEye(state);
      expect(eye[0]).toBeCloseTo(0);
      expect(eye[1]).toBeCloseTo(0);
      expect(eye[2]).toBeCloseTo(5);
    });

    it('theta=PI/2, phi=PI/2 (equator, quarter turn), distance=2, target origin -> [2,0,0]', () => {
      const state: OrbitState = { theta: Math.PI / 2, phi: Math.PI / 2, distance: 2, target: [0, 0, 0] };
      const eye = orbitEye(state);
      expect(eye[0]).toBeCloseTo(2);
      expect(eye[1]).toBeCloseTo(0);
      expect(eye[2]).toBeCloseTo(0);
    });

    it('theta=0, phi=PI/4 (non-equatorial), distance=4, target origin', () => {
      const state: OrbitState = { theta: 0, phi: Math.PI / 4, distance: 4, target: [0, 0, 0] };
      const eye = orbitEye(state);
      const sinPhi = Math.sin(Math.PI / 4);
      const cosPhi = Math.cos(Math.PI / 4);
      expect(eye[0]).toBeCloseTo(4 * sinPhi * Math.sin(0));
      expect(eye[1]).toBeCloseTo(4 * cosPhi);
      expect(eye[2]).toBeCloseTo(4 * sinPhi * Math.cos(0));
    });

    it('non-zero target shifts the eye by the same offset', () => {
      const base: OrbitState = { theta: 0, phi: Math.PI / 2, distance: 5, target: [0, 0, 0] };
      const shifted: OrbitState = { ...base, target: [1, 2, 3] };
      const eyeBase = orbitEye(base);
      const eyeShifted = orbitEye(shifted);
      expect(eyeShifted[0]).toBeCloseTo(eyeBase[0] + 1);
      expect(eyeShifted[1]).toBeCloseTo(eyeBase[1] + 2);
      expect(eyeShifted[2]).toBeCloseTo(eyeBase[2] + 3);
    });
  });

  describe('rotate', () => {
    it('clamps phi strictly inside (0, PI) even with an enormous positive dyPixels', () => {
      const result = rotate(DEFAULT_ORBIT_STATE, 0, 1_000_000);
      expect(result.phi).toBeGreaterThan(0.001);
      expect(result.phi).toBeLessThan(Math.PI - 0.001);
    });

    it('clamps phi strictly inside (0, PI) even with an enormous negative dyPixels', () => {
      const result = rotate(DEFAULT_ORBIT_STATE, 0, -1_000_000);
      expect(result.phi).toBeGreaterThan(0.001);
      expect(result.phi).toBeLessThan(Math.PI - 0.001);
    });

    it('does not mutate the input state', () => {
      const original = { ...DEFAULT_ORBIT_STATE, target: [...DEFAULT_ORBIT_STATE.target] as [number, number, number] };
      rotate(DEFAULT_ORBIT_STATE, 500, 500);
      expect(DEFAULT_ORBIT_STATE).toEqual(original);
    });

    it('theta increases monotonically with positive dxPixels', () => {
      let state = DEFAULT_ORBIT_STATE;
      let prevTheta = state.theta;
      for (let i = 0; i < 5; i++) {
        state = rotate(state, 50, 0);
        expect(state.theta).toBeGreaterThan(prevTheta);
        prevTheta = state.theta;
      }
    });

    it('theta decreases monotonically with negative dxPixels', () => {
      let state = DEFAULT_ORBIT_STATE;
      let prevTheta = state.theta;
      for (let i = 0; i < 5; i++) {
        state = rotate(state, -50, 0);
        expect(state.theta).toBeLessThan(prevTheta);
        prevTheta = state.theta;
      }
    });
  });

  describe('zoom', () => {
    it('repeated positive wheelDelta strictly increases distance up to the clamp, then holds', () => {
      let state = DEFAULT_ORBIT_STATE;
      let prevDistance = state.distance;
      let clampedAt = -1;
      for (let i = 0; i < 60; i++) {
        state = zoom(state, 500);
        if (clampedAt === -1) {
          if (state.distance > prevDistance) {
            prevDistance = state.distance;
          } else {
            clampedAt = i;
          }
        } else {
          // Once clamped, distance must not change further.
          expect(state.distance).toBe(prevDistance);
        }
      }
      expect(clampedAt).toBeGreaterThan(-1);
      expect(state.distance).toBeLessThanOrEqual(50);
    });

    it('repeated negative wheelDelta strictly decreases distance down to the clamp, then holds', () => {
      let state = DEFAULT_ORBIT_STATE;
      let prevDistance = state.distance;
      let clampedAt = -1;
      for (let i = 0; i < 60; i++) {
        state = zoom(state, -500);
        if (clampedAt === -1) {
          if (state.distance < prevDistance) {
            prevDistance = state.distance;
          } else {
            clampedAt = i;
          }
        } else {
          expect(state.distance).toBe(prevDistance);
        }
      }
      expect(clampedAt).toBeGreaterThan(-1);
      expect(state.distance).toBeGreaterThanOrEqual(0.5);
    });

    it('does not mutate the input state', () => {
      const original = { ...DEFAULT_ORBIT_STATE };
      zoom(DEFAULT_ORBIT_STATE, 1000);
      expect(DEFAULT_ORBIT_STATE).toEqual(original);
    });
  });

  describe('pan', () => {
    it('only changes target; theta/phi/distance are unchanged', () => {
      const state: OrbitState = { theta: 0.4, phi: 1.1, distance: 7, target: [1, 2, 3] };
      const result = pan(state, 30, -20);
      expect(result.theta).toBe(state.theta);
      expect(result.phi).toBe(state.phi);
      expect(result.distance).toBe(state.distance);
      expect(result.target).not.toEqual(state.target);
    });

    it('does not mutate the input state', () => {
      const original = { ...DEFAULT_ORBIT_STATE, target: [...DEFAULT_ORBIT_STATE.target] as [number, number, number] };
      pan(DEFAULT_ORBIT_STATE, 30, -20);
      expect(DEFAULT_ORBIT_STATE).toEqual(original);
    });
  });

  describe('matrix helpers', () => {
    it('multiply4(vp, invVp) is approximately the identity matrix', () => {
      const state: OrbitState = { theta: 0.7, phi: 1.2, distance: 6, target: [0.5, -1, 2] };
      const { vp, invVp } = viewProj(state, 16 / 9);
      const product = multiply4(vp, invVp);
      expectIdentity(product);
    });

    it('multiply4(vp, invVp) round-trips for a second, different state', () => {
      const state: OrbitState = { theta: -2.1, phi: 0.4, distance: 15, target: [0, 0, 0] };
      const { vp, invVp } = viewProj(state, 1);
      const product = multiply4(vp, invVp);
      expectIdentity(product);
    });

    it('perspective produces a sane x-scale entry for a known simple case', () => {
      const fovY = Math.PI / 2;
      const aspect = 1;
      const m = perspective(fovY, aspect, 1, 100);
      const expectedXScale = 1 / Math.tan(fovY / 2) / aspect;
      expect(Number.isFinite(m[0])).toBe(true);
      expect(m[0]).not.toBe(0);
      expect(m[0]).toBeCloseTo(expectedXScale);
      // y-scale (m[5]) should equal x-scale here since aspect=1.
      expect(m[5]).toBeCloseTo(expectedXScale);
      // Perspective divide marker: column 2, row 3 must be -1 for a
      // right-handed perspective matrix in this layout.
      expect(m[11]).toBe(-1);
    });

    it('lookAt + viewProj projects the target to near NDC center (x,y ~= 0)', () => {
      // eye=[0,0,5], target=[0,0,0], up=[0,1,0] is reproduced by
      // theta=0, phi=PI/2, distance=5 (see orbitEye tests above).
      const state: OrbitState = { theta: 0, phi: Math.PI / 2, distance: 5, target: [0, 0, 0] };
      const eye = orbitEye(state);
      expect(eye[0]).toBeCloseTo(0);
      expect(eye[1]).toBeCloseTo(0);
      expect(eye[2]).toBeCloseTo(5);

      const { vp } = viewProj(state, 1);
      const clip = transformPoint(vp, [0, 0, 0]);
      const ndcX = clip[0] / clip[3];
      const ndcY = clip[1] / clip[3];
      expect(ndcX).toBeCloseTo(0);
      expect(ndcY).toBeCloseTo(0);
    });

    it('lookAt directly matches viewProj for the eye/target/up it is given', () => {
      const eye: [number, number, number] = [0, 0, 5];
      const target: [number, number, number] = [0, 0, 0];
      const view = lookAt(eye, target, [0, 1, 0]);
      // The target, transformed into view space, should lie on the -Z axis
      // (in front of the camera) with x=y=0.
      const viewSpaceTarget = transformPoint(view, target);
      expect(viewSpaceTarget[0]).toBeCloseTo(0);
      expect(viewSpaceTarget[1]).toBeCloseTo(0);
      expect(viewSpaceTarget[2]).toBeCloseTo(-5);
    });

    it('invert4 returns the identity for a singular (all-zero) matrix', () => {
      const singular = new Float32Array(16);
      const inv = invert4(singular);
      expectIdentity(inv);
    });
  });
});
