/**
 * VolumeRenderer — a hand-written WebGL2 volume raycaster for the 3D tab.
 * Renders two co-registered 3-D textures at once (a grayscale raw-intensity
 * volume and a class-index label volume), composited with per-class colors
 * and independent opacity controls, orbited/panned/zoomed with an arcball-
 * style camera. No three.js/gl-matrix — camera math lives in `orbitCamera.ts`
 * (imported, never reimplemented here), matching this repo's style of
 * hand-rolling things like livewire/magicwand/CLAHE elsewhere in
 * `frontend/src/lib/`.
 *
 * Deliberately untested — jsdom has no WebGL2 context, so any mock would
 * assert nothing real (a mocked `gl.texSubImage3D` can't tell you the
 * texture upload was byte-correct, and a mocked shader compile can't tell
 * you the GLSL is valid). Testable math (camera, dims, label rasterization)
 * lives in `orbitCamera.ts` / `volumeDims.ts` / `labelVolume.ts`, which do
 * have unit tests. This file's correctness has to be checked by eye in a
 * real browser (the "3D" tab) and by `tsc` for type errors — see
 * `package.json`'s `typecheck` script.
 */
import {
  DEFAULT_ORBIT_STATE,
  pan as panOrbit,
  rotate as rotateOrbit,
  viewProj,
  zoom as zoomOrbit,
  type OrbitState,
} from './orbitCamera';
import { FRAGMENT_SHADER_SOURCE, VERTEX_SHADER_SOURCE } from './shaders';
import type { VolumeDims } from './volumeDims';

export type { VolumeDims };

export type VolumeMode = 'both' | 'raw' | 'labels';

export interface VolumeRenderOptions {
  rawOpacity: number; // 0..1
  labelOpacity: number; // 0..1
  windowLo: number; // 0..1
  windowHi: number; // 0..1, must be > windowLo
  mode: VolumeMode;
  zScale: number; // voxel aspect correction, e.g. 0.2..5
  quality: 'interactive' | 'full'; // fewer raymarch steps while dragging
  /** Gradient-lit ("realistic") shading vs. the default flat look — see
   *  `shaders.ts`'s `rawGradient` doc comment. Off by default: it costs 6
   *  extra raw-texture fetches per raymarch step, so it's opt-in rather than
   *  always-on. */
  shading: boolean;
  /** 0..1 — how strongly `shading` modulates color once it's on (see
   *  `shaders.ts`'s uShading block). Only meaningful when `shading` is true;
   *  0 is indistinguishable from `shading: false` other than the extra cost. */
  shadingStrength: number;
}

const DEFAULT_OPTIONS: VolumeRenderOptions = {
  rawOpacity: 1,
  labelOpacity: 0.6,
  windowLo: 0,
  windowHi: 1,
  mode: 'both',
  zScale: 1,
  quality: 'full',
  shading: false,
  shadingStrength: 0.8,
};

/** Raymarch step counts for each quality tier — see `shaders.ts`'s
 *  `correctionExponent` comment for why changing this doesn't also change
 *  the volume's apparent density. */
const STEPS_INTERACTIVE = 128;
const STEPS_FULL = 384;

/** `uMode` shader values, per `shaders.ts`'s uniform doc comment. */
const MODE_TO_INT: Record<VolumeMode, number> = { both: 0, raw: 1, labels: 2 };

/** A trivial 1x1x1 all-zero volume, used both as the label texture's "no
 *  data" fallback (the shader never needs a null-texture branch) and as the
 *  raw texture's placeholder before the caller's first `setRawVolume` call,
 *  so `draw()` is always safe to call from the moment the constructor
 *  returns. */
const TRIVIAL_DIMS: VolumeDims = { nz: 1, ny: 1, nx: 1, sxy: 1, sz: 1 };
const TRIVIAL_DATA = new Uint8Array([0]);

/** All-zero 256x1 RGBA LUT (every class fully transparent) — the LUT
 *  placeholder before the caller's first `setClassLUT` call. */
const TRIVIAL_LUT = new Uint8Array(1024);

const UNIFORM_NAMES = [
  'uRawTex',
  'uLabelTex',
  'uLutTex',
  'uInvViewProj',
  'uEye',
  'uBoxHalf',
  'uWindow',
  'uRawOpacity',
  'uLabelOpacity',
  'uMode',
  'uSteps',
  'uShading',
  'uTexelSize',
  'uLightDir',
  'uShadingStrength',
] as const;

type UniformName = (typeof UNIFORM_NAMES)[number];

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export class VolumeRendererError extends Error {
  constructor(
    public readonly code: 'webgl2-unavailable' | 'shader-compile-failed' | 'program-link-failed',
    message: string
  ) {
    super(message);
    this.name = 'VolumeRendererError';
  }
}

export class VolumeRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;

  private program: WebGLProgram | null = null;
  private uniforms: Record<UniformName, WebGLUniformLocation | null> = Object.fromEntries(
    UNIFORM_NAMES.map((n) => [n, null])
  ) as Record<UniformName, WebGLUniformLocation | null>;

  private rawTex: WebGLTexture | null = null;
  private labelTex: WebGLTexture | null = null;
  private lutTex: WebGLTexture | null = null;

  // Retained inputs, replayed into fresh GL resources after a context-loss ->
  // context-restore cycle so the caller never has to re-call the setters.
  private rawData: Uint8Array = TRIVIAL_DATA;
  private rawDims: VolumeDims = TRIVIAL_DIMS;
  private labelData: Uint8Array | null = null; // null => render the trivial fallback
  private labelDims: VolumeDims = TRIVIAL_DIMS;
  private lut: Uint8Array = TRIVIAL_LUT;

  private orbitState: OrbitState = { ...DEFAULT_ORBIT_STATE, target: [...DEFAULT_ORBIT_STATE.target] };
  private options: VolumeRenderOptions = { ...DEFAULT_OPTIONS };

  private dirty = false;
  private rafHandle: number | null = null;

  private readonly handleContextLost: (event: Event) => void;
  private readonly handleContextRestored: () => void;

  /**
   * @throws VolumeRendererError('webgl2-unavailable') if the canvas can't
   *   produce a WebGL2 context.
   * @throws VolumeRendererError('shader-compile-failed' | 'program-link-failed')
   *   if the shaders in `shaders.ts` fail to compile/link (should only
   *   happen from a genuine GLSL bug or an unusually old/broken driver — the
   *   message includes the driver's own info log).
   */
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new VolumeRendererError('webgl2-unavailable', 'WebGL2 is not available on this canvas/browser.');
    }
    this.gl = gl;

    // `preventDefault()` on context-lost tells the browser we intend to
    // recover (via `handleContextRestored`) rather than treating the canvas
    // as permanently dead.
    this.handleContextLost = (event: Event) => {
      event.preventDefault();
      this.program = null;
      this.rawTex = null;
      this.labelTex = null;
      this.lutTex = null;
      if (this.rafHandle !== null) {
        cancelAnimationFrame(this.rafHandle);
        this.rafHandle = null;
      }
    };
    this.handleContextRestored = () => {
      // Recompiles the program and recreates all textures from the retained
      // rawData/rawDims/labelData/labelDims/lut fields — the caller does not
      // need to call setRawVolume/setLabelVolume/setClassLUT again.
      this.setupGl();
      this.markDirty();
    };
    canvas.addEventListener('webglcontextlost', this.handleContextLost, false);
    canvas.addEventListener('webglcontextrestored', this.handleContextRestored, false);

    this.setupGl();
  }

  // ---------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------

  /**
   * Upload a new raw intensity volume. `data.length` must equal
   * `dims.nx * dims.ny * dims.nz`, laid out z-major/y-middle/x-fastest
   * (matching `useVolume.ts`'s `/api/image/volume` payload and
   * `gl.texSubImage3D`'s own expected data order — depth slowest, width
   * fastest — so no reshuffling is needed before upload).
   */
  setRawVolume(data: Uint8Array, dims: VolumeDims): void {
    this.warnIfSizeMismatch('setRawVolume', data, dims);
    this.rawData = data;
    this.rawDims = dims;
    if (this.program) {
      this.uploadVolumeTexture('raw', data, dims);
    }
    this.markDirty();
  }

  /**
   * Upload a new label volume, or clear it. When `data` is `null`, a 1x1x1
   * all-zero texture is bound instead (voxel value 0 = background, which the
   * LUT maps to alpha 0), so the fragment shader never needs a null-texture
   * branch. `dims` is still required in that case only insofar as the type
   * signature demands it; it's ignored for the trivial-texture path.
   */
  setLabelVolume(data: Uint8Array | null, dims: VolumeDims): void {
    if (data) this.warnIfSizeMismatch('setLabelVolume', data, dims);
    this.labelData = data;
    this.labelDims = dims;
    if (this.program) {
      if (data) this.uploadVolumeTexture('label', data, dims);
      else this.uploadVolumeTexture('label', TRIVIAL_DATA, TRIVIAL_DIMS);
    }
    this.markDirty();
  }

  /** `lut` must be length 1024 (256 RGBA entries): index 0 = background
   *  (should be fully transparent), index i = class (i-1)'s display color
   *  (alpha 0 when that class is hidden). Silently ignored (with a console
   *  warning) if the length is wrong, rather than uploading a
   *  misinterpreted/truncated texture. */
  setClassLUT(lut: Uint8Array): void {
    if (lut.length !== 1024) {
      console.warn(
        `VolumeRenderer.setClassLUT: expected a 1024-byte (256 * RGBA) lookup table, got ${lut.length} bytes; ignoring.`
      );
      return;
    }
    this.lut = lut;
    if (this.program) this.uploadLut(lut);
    this.markDirty();
  }

  /** Shallow-merges `opts` into the current options. Window bounds are
   *  clamped defensively (the caller is expected to keep windowHi > windowLo,
   *  but a degenerate window would otherwise divide by ~0 in the shader and
   *  render something confusingly wrong rather than obviously broken). */
  setOptions(opts: Partial<VolumeRenderOptions>): void {
    this.options = { ...this.options, ...opts };
    this.options.rawOpacity = clamp01(this.options.rawOpacity);
    this.options.labelOpacity = clamp01(this.options.labelOpacity);
    this.options.windowLo = clamp01(this.options.windowLo);
    this.options.windowHi = clamp01(this.options.windowHi);
    if (this.options.windowHi <= this.options.windowLo) {
      this.options.windowHi = Math.min(1, this.options.windowLo + 1e-3);
    }
    this.options.shadingStrength = clamp01(this.options.shadingStrength);
    this.markDirty();
  }

  /** Orbit the camera by a raw pointer-move delta in pixels. */
  orbit(dxPixels: number, dyPixels: number): void {
    this.orbitState = rotateOrbit(this.orbitState, dxPixels, dyPixels);
    this.markDirty();
  }

  /** Dolly-zoom by a raw wheel `deltaY`. */
  zoomBy(wheelDelta: number): void {
    this.orbitState = zoomOrbit(this.orbitState, wheelDelta);
    this.markDirty();
  }

  /** Pan the orbit target by a raw pointer-move delta in pixels. */
  panBy(dxPixels: number, dyPixels: number): void {
    this.orbitState = panOrbit(this.orbitState, dxPixels, dyPixels);
    this.markDirty();
  }

  /** Reset the camera to `orbitCamera`'s default orbit state. */
  resetView(): void {
    this.orbitState = { ...DEFAULT_ORBIT_STATE, target: [...DEFAULT_ORBIT_STATE.target] };
    this.markDirty();
  }

  /** Sync the canvas's backing-store size to its CSS/layout size (capped at
   *  devicePixelRatio 2, so a 4K/5K display doesn't force absurdly expensive
   *  raymarching), and update the GL viewport. Callers typically wire this to
   *  a `ResizeObserver` on the canvas's parent (see `VolumeCanvas.tsx`). */
  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.gl.viewport(0, 0, width, height);
      this.markDirty();
    }
  }

  /** Release all GL resources and event listeners. The instance is unusable
   *  afterward — construct a new one if needed. */
  dispose(): void {
    const gl = this.gl;
    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored);
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    if (this.rawTex) gl.deleteTexture(this.rawTex);
    if (this.labelTex) gl.deleteTexture(this.labelTex);
    if (this.lutTex) gl.deleteTexture(this.lutTex);
    if (this.program) gl.deleteProgram(this.program);
    this.rawTex = null;
    this.labelTex = null;
    this.lutTex = null;
    this.program = null;
  }

  // ---------------------------------------------------------------------
  // Render scheduling — on-demand, never a continuous rAF loop.
  // ---------------------------------------------------------------------

  private markDirty(): void {
    this.dirty = true;
    if (this.rafHandle !== null) return; // a frame is already pending
    this.rafHandle = requestAnimationFrame(() => {
      this.rafHandle = null;
      if (!this.dirty) return;
      this.dirty = false;
      this.draw();
    });
  }

  // ---------------------------------------------------------------------
  // GL setup (initial construction AND context-restore both funnel through
  // here, so the two paths can never drift apart).
  // ---------------------------------------------------------------------

  private setupGl(): void {
    const gl = this.gl;

    this.program = this.compileProgram();
    gl.useProgram(this.program);
    this.uniforms = this.cacheUniformLocations(this.program);

    // Texture units are fixed for the program's whole lifetime.
    gl.uniform1i(this.uniforms.uRawTex, 0);
    gl.uniform1i(this.uniforms.uLabelTex, 1);
    gl.uniform1i(this.uniforms.uLutTex, 2);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.clearColor(0, 0, 0, 0);

    // Recreate textures from whatever was last retained (real caller data,
    // or this class's own trivial placeholders on first construction).
    this.uploadVolumeTexture('raw', this.rawData, this.rawDims);
    this.uploadVolumeTexture('label', this.labelData ?? TRIVIAL_DATA, this.labelData ? this.labelDims : TRIVIAL_DIMS);
    this.uploadLut(this.lut);
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) {
      throw new VolumeRendererError('shader-compile-failed', 'gl.createShader returned null (context lost?).');
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader) ?? '(no info log)';
      gl.deleteShader(shader);
      const kind = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
      throw new VolumeRendererError('shader-compile-failed', `Failed to compile ${kind} shader:\n${info}`);
    }
    return shader;
  }

  private compileProgram(): WebGLProgram {
    const gl = this.gl;
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

    const program = gl.createProgram();
    if (!program) {
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      throw new VolumeRendererError('program-link-failed', 'gl.createProgram returned null (context lost?).');
    }
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    // Flags the shaders for deletion once detached; the program already
    // holds what it needs post-link, so this ordering is safe.
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program) ?? '(no info log)';
      gl.deleteProgram(program);
      throw new VolumeRendererError('program-link-failed', `Failed to link volume raycaster program:\n${info}`);
    }
    return program;
  }

  private cacheUniformLocations(program: WebGLProgram): Record<UniformName, WebGLUniformLocation | null> {
    const gl = this.gl;
    const out = {} as Record<UniformName, WebGLUniformLocation | null>;
    for (const name of UNIFORM_NAMES) {
      out[name] = gl.getUniformLocation(program, name);
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Texture upload
  // ---------------------------------------------------------------------

  private warnIfSizeMismatch(caller: string, data: Uint8Array, dims: VolumeDims): void {
    const expected = dims.nx * dims.ny * dims.nz;
    if (data.length !== expected) {
      console.warn(
        `VolumeRenderer.${caller}: data.length (${data.length}) does not match ` +
          `dims.nx*ny*nz (${dims.nx}*${dims.ny}*${dims.nz} = ${expected}); the upload will likely ` +
          'be misaligned or throw.'
      );
    }
  }

  /** (Re)creates the raw or label 3-D texture and uploads `data` into it.
   *  Textures are immutable-storage (`texStorage3D`), and dims can change
   *  between calls (e.g. a new dataset), so the old texture is deleted and a
   *  fresh one allocated each time rather than trying to resize storage
   *  in place. */
  private uploadVolumeTexture(which: 'raw' | 'label', data: Uint8Array, dims: VolumeDims): void {
    const gl = this.gl;
    const filter = which === 'raw' ? gl.LINEAR : gl.NEAREST;

    const texture = gl.createTexture();
    if (!texture) return; // context lost mid-call; the restore handler will redo this

    gl.bindTexture(gl.TEXTURE_3D, texture);
    // CRITICAL: R8 data whose row length (nx bytes) isn't a multiple of 4 is
    // silently corrupted by the default 4-byte UNPACK_ALIGNMENT (each row
    // after the first starts up to 3 bytes early/late). Must be set before
    // every upload, not just once — it is texture-upload-call state, not
    // texture-object state.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, Math.max(1, dims.nx), Math.max(1, dims.ny), Math.max(1, dims.nz));
    gl.texSubImage3D(
      gl.TEXTURE_3D,
      0,
      0,
      0,
      0,
      Math.max(1, dims.nx),
      Math.max(1, dims.ny),
      Math.max(1, dims.nz),
      gl.RED,
      gl.UNSIGNED_BYTE,
      data
    );
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

    if (which === 'raw') {
      if (this.rawTex) gl.deleteTexture(this.rawTex);
      this.rawTex = texture;
    } else {
      if (this.labelTex) gl.deleteTexture(this.labelTex);
      this.labelTex = texture;
    }
  }

  private uploadLut(lut: Uint8Array): void {
    const gl = this.gl;
    if (!this.lutTex) this.lutTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // ---------------------------------------------------------------------
  // Draw
  // ---------------------------------------------------------------------

  /**
   * Half-extents of the volume's world-space bounding box, aspect-corrected
   * for voxel dimensions and `zScale`, normalized so the largest axis has
   * half-extent 0.5 (i.e. the box's longest side is exactly 1 world unit).
   * `orbitCamera.DEFAULT_ORBIT_STATE.distance` (3) is chosen to comfortably
   * frame a box of roughly this size — see that file's own doc comment.
   */
  private computeBoxHalf(): [number, number, number] {
    const { nx, ny, nz } = this.rawDims;
    const dz = nz * this.options.zScale;
    const maxDim = Math.max(nx, ny, dz, 1e-6);
    return [nx / maxDim / 2, ny / maxDim / 2, dz / maxDim / 2];
  }

  /**
   * World-space unit "surface -> light" direction for the gradient-shading
   * key light, recomputed every frame from the camera's own basis (offset
   * up-and-to-the-side of the eye, NOT coincident with the view direction —
   * see `shaders.ts`'s uShading doc comment for why an on-axis "headlamp"
   * light reads as nearly flat). Deriving it from `forward`/`right`/`up`
   * rather than a fixed world-space constant is what makes the shading track
   * the camera as the user orbits, instead of sliding across the volume.
   */
  private computeKeyLightDir(eye: readonly [number, number, number]): [number, number, number] {
    const target = this.orbitState.target;
    const fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
    const fLen = Math.hypot(fx, fy, fz) || 1;
    const forward: [number, number, number] = [fx / fLen, fy / fLen, fz / fLen];

    // orbitCamera keeps `phi` off the poles, so `forward` should never be
    // exactly parallel to world-up — guarded anyway rather than ever risking
    // a zero-length `right` vector.
    const upHint: [number, number, number] = Math.abs(forward[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];

    const rx = forward[1] * upHint[2] - forward[2] * upHint[1];
    const ry = forward[2] * upHint[0] - forward[0] * upHint[2];
    const rz = forward[0] * upHint[1] - forward[1] * upHint[0];
    const rLen = Math.hypot(rx, ry, rz) || 1;
    const right: [number, number, number] = [rx / rLen, ry / rLen, rz / rLen];

    const upx = right[1] * forward[2] - right[2] * forward[1];
    const upy = right[2] * forward[0] - right[0] * forward[2];
    const upz = right[0] * forward[1] - right[1] * forward[0];

    // Bias toward the eye (-forward), up, and slightly to one side — an
    // upper-side key light, the classic non-flat portrait-lighting angle.
    const lx = -forward[0] * 0.8 + upx * 0.55 - right[0] * 0.35;
    const ly = -forward[1] * 0.8 + upy * 0.55 - right[1] * 0.35;
    const lz = -forward[2] * 0.8 + upz * 0.55 - right[2] * 0.35;
    const lLen = Math.hypot(lx, ly, lz) || 1;
    return [lx / lLen, ly / lLen, lz / lLen];
  }

  private draw(): void {
    const gl = this.gl;
    const program = this.program;
    // Context lost (program cleared) and not yet restored — nothing to draw;
    // the restore handler will schedule a fresh frame once it's back.
    if (!program) return;

    const width = this.canvas.width || 1;
    const height = this.canvas.height || 1;
    const aspect = width / height;

    const { invVp, eye } = viewProj(this.orbitState, aspect);
    const boxHalf = this.computeBoxHalf();
    const steps = this.options.quality === 'interactive' ? STEPS_INTERACTIVE : STEPS_FULL;
    const modeIndex = MODE_TO_INT[this.options.mode];
    // Cheap even when shading is off — a few dozen scalar ops, not worth an
    // `if (this.options.shading)` branch to skip.
    const lightDir = this.computeKeyLightDir(eye);

    gl.viewport(0, 0, width, height);
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.rawTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.labelTex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);

    gl.uniformMatrix4fv(this.uniforms.uInvViewProj, false, invVp);
    gl.uniform3fv(this.uniforms.uEye, eye);
    gl.uniform3fv(this.uniforms.uBoxHalf, boxHalf);
    gl.uniform2f(this.uniforms.uWindow, this.options.windowLo, this.options.windowHi);
    gl.uniform1f(this.uniforms.uRawOpacity, this.options.rawOpacity);
    gl.uniform1f(this.uniforms.uLabelOpacity, this.options.labelOpacity);
    gl.uniform1i(this.uniforms.uMode, modeIndex);
    gl.uniform1i(this.uniforms.uSteps, steps);
    gl.uniform1i(this.uniforms.uShading, this.options.shading ? 1 : 0);
    gl.uniform3f(
      this.uniforms.uTexelSize,
      1 / Math.max(1, this.rawDims.nx),
      1 / Math.max(1, this.rawDims.ny),
      1 / Math.max(1, this.rawDims.nz)
    );
    gl.uniform3fv(this.uniforms.uLightDir, lightDir);
    gl.uniform1f(this.uniforms.uShadingStrength, this.options.shadingStrength);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
