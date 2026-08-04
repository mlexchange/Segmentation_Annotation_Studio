/**
 * GLSL ES 3.00 shader sources for the hand-written WebGL2 volume raycaster
 * (see `volumeRenderer.ts`). No three.js/gl-matrix and no GLSL-bundler Vite
 * plugin are installed in this repo, so shader source just lives here as
 * plain exported template strings, the same "hand-roll it, no extra
 * tooling" style as `livewire.ts`/`magicwand.ts`/`clahe.ts` elsewhere in
 * `frontend/src/lib/`.
 *
 * IMPORTANT: `#version 300 es` must be the literal first line of each source
 * string below (no leading blank line or comment) — WebGL2's shader
 * compiler rejects the version pragma if anything precedes it, including
 * whitespace right after the template literal's opening backtick.
 */

/**
 * Fullscreen-triangle vertex shader with NO vertex buffer and NO attributes:
 * a single oversized triangle is generated purely from `gl_VertexID` (the
 * classic "0,1,2 -> screen-covering triangle" trick), so the entire draw
 * call is `gl.drawArrays(gl.TRIANGLES, 0, 3)` with nothing bound — no VAO,
 * no VBO.
 *
 * `pos` takes the values (0,0), (2,0), (0,2) for vertex ids 0,1,2. After the
 * `*2.0 - 1.0` remap that's NDC (-1,-1), (3,-1), (-1,3): a triangle that
 * fully covers the visible (-1,1) NDC square and extends past it (the extra
 * area is simply clipped by the GPU), with no diagonal seam the way two
 * triangles forming a quad would have.
 *
 * `vNdc` is forwarded to the fragment shader so it can reconstruct the
 * exact NDC coordinate of each pixel without a `uResolution` uniform or
 * `gl_FragCoord`/viewport arithmetic.
 */
export const VERTEX_SHADER_SOURCE = `#version 300 es
out vec2 vNdc;

void main() {
  vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vNdc = pos * 2.0 - 1.0;
  gl_Position = vec4(vNdc, 0.0, 1.0);
}
`;

/**
 * Raymarching fragment shader. Composites two co-registered 3-D textures —
 * a grayscale raw-intensity volume and a class-index label volume — front
 * to back along a view ray through an axis-aligned box in world space, then
 * writes the accumulated color/alpha.
 *
 * Non-obvious conventions (read before touching):
 *
 * - The box is centered on the world origin and spans [-uBoxHalf, +uBoxHalf]
 *   per axis. `uBoxHalf` already encodes the volume's voxel-grid aspect
 *   ratio (see `volumeRenderer.ts`'s `computeBoxHalf`) — this shader doesn't
 *   need nx/ny/nz itself, only the box it's marching through.
 * - World -> texture coordinates: `tc = p / (2*uBoxHalf) + 0.5` maps the box
 *   into [0,1]^3. The Y axis is then FLIPPED (`tc.y = 1.0 - tc.y`): the
 *   raw/label 3-D textures are uploaded with texture row 0 = the first row
 *   of the source image (the top of the flat 2-D slice view used elsewhere
 *   in this app), but `orbitCamera`'s world space is Y-up — without the
 *   flip the volume would render upside-down relative to the 2-D slice
 *   viewer the user is comparing it against. The Z (slice) axis is NOT
 *   flipped; texture w increases with slice index, matching upload order in
 *   `volumeRenderer.ts`'s `setRawVolume`/`setLabelVolume`.
 * - The ray direction is computed as `normalize(unprojectedFarPoint - uEye)`
 *   rather than unprojecting two NDC depths and subtracting two unprojected
 *   points: `uEye` already IS the exact camera position (it comes straight
 *   from `orbitCamera.viewProj`), so only one unproject is needed.
 */
export const FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
precision highp sampler2D;

in vec2 vNdc;
out vec4 outColor;

uniform sampler3D uRawTex;     // R8, LINEAR filtering
uniform sampler3D uLabelTex;   // R8, NEAREST filtering — stores classIndex/255
uniform sampler2D uLutTex;     // 256x1 RGBA8; index 0 = background (transparent)
uniform mat4 uInvViewProj;
uniform vec3 uEye;
uniform vec3 uBoxHalf;
uniform vec2 uWindow;          // (lo, hi) intensity window in [0,1]
uniform float uRawOpacity;     // 0..1
uniform float uLabelOpacity;   // 0..1
uniform int uMode;             // 0 = both, 1 = raw only, 2 = labels only
uniform int uSteps;            // raymarch sample count
uniform int uShading;          // 0 = flat (default), 1 = gradient-lit ("realistic")
uniform vec3 uTexelSize;       // (1/nx, 1/ny, 1/nz) of the raw volume texture
uniform vec3 uLightDir;        // world-space surface->light unit vector (camera-relative key light)
uniform float uShadingStrength; // 0..1 — how strongly shading modulates color (see uShading block below)

/** Unproject an NDC point (xy in [-1,1], z the clip-space depth in [-1,1])
 *  through the inverse view-projection matrix into a world-space point. */
vec3 unprojectNdc(vec2 ndc, float z) {
  vec4 clip = vec4(ndc, z, 1.0);
  vec4 world = uInvViewProj * clip;
  return world.xyz / world.w;
}

/**
 * Approximate a surface normal from the LOCAL GRADIENT of the raw intensity
 * field at texture coordinate 'tc', via a central-difference sample on each
 * axis (the classic cheap volume-shading normal — no precomputed gradient
 * texture, just 6 extra 'uRawTex' fetches, only ever paid when 'uShading'
 * is on).
 *
 * The raw finite difference is in TEXTURE-space units; it's rescaled into
 * WORLD-space units (dividing by '2*uBoxHalf', the world extent one texture
 * axis spans) before use, so anisotropic voxel spacing — 'uBoxHalf.z' moves
 * with the user's 'zScale' control, which can range 0.2-5x — doesn't skew
 * the normal's DIRECTION the way a naive texture-space gradient would.
 *
 * The Y component is negated to correct for the same texture Y-flip
 * documented on the module comment above ('tc.y = 1 - worldMappedY', so
 * texture-space and world-space Y move in opposite directions).
 *
 * Returns a non-normalized vector; callers must check its length before
 * normalizing (a perfectly homogeneous region has zero gradient, and
 * 'normalize(vec3(0))' is undefined).
 */
vec3 rawGradient(vec3 tc) {
  float gx = texture(uRawTex, tc + vec3(uTexelSize.x, 0.0, 0.0)).r
           - texture(uRawTex, tc - vec3(uTexelSize.x, 0.0, 0.0)).r;
  float gy = texture(uRawTex, tc + vec3(0.0, uTexelSize.y, 0.0)).r
           - texture(uRawTex, tc - vec3(0.0, uTexelSize.y, 0.0)).r;
  float gz = texture(uRawTex, tc + vec3(0.0, 0.0, uTexelSize.z)).r
           - texture(uRawTex, tc - vec3(0.0, 0.0, uTexelSize.z)).r;
  vec3 worldScale = 1.0 / max(2.0 * uBoxHalf, vec3(1e-6));
  return vec3(gx, -gy, gz) * worldScale;
}

void main() {
  vec3 rayOrigin = uEye;
  vec3 farPoint = unprojectNdc(vNdc, 1.0);
  vec3 rayDir = normalize(farPoint - rayOrigin);

  // Guard exactly-zero direction components before the slab division below.
  // IEEE-754 division by +-0 yields +-inf, which min()/max() then propagate
  // into a correct finite tNear/tFar for an axis-aligned ray — but not every
  // GPU driver's "fast math" mode honors that reliably (some produce NaN
  // instead, which poisons every later min/max). Nudging a vanishingly-small
  // direction component off exact zero costs nothing at this precision and
  // sidesteps driver-dependent behavior entirely.
  if (abs(rayDir.x) < 1e-8) rayDir.x = 1e-8;
  if (abs(rayDir.y) < 1e-8) rayDir.y = 1e-8;
  if (abs(rayDir.z) < 1e-8) rayDir.z = 1e-8;

  // Ray-box slab intersection against [-uBoxHalf, +uBoxHalf].
  vec3 invDir = 1.0 / rayDir;
  vec3 boxMin = -uBoxHalf;
  vec3 boxMax = uBoxHalf;
  vec3 t1 = (boxMin - rayOrigin) * invDir;
  vec3 t2 = (boxMax - rayOrigin) * invDir;
  vec3 tMinV = min(t1, t2);
  vec3 tMaxV = max(t1, t2);
  float tNear = max(max(tMinV.x, tMinV.y), tMinV.z);
  float tFar = min(min(tMaxV.x, tMaxV.y), tMaxV.z);

  // Never march behind the camera (relevant when the eye is inside the box,
  // where the raw tNear would be negative).
  tNear = max(tNear, 0.0);

  // Ray misses the box entirely, or the box is entirely behind the camera.
  if (tNear >= tFar) {
    outColor = vec4(0.0);
    return;
  }

  int steps = max(uSteps, 1);
  float stepSize = (tFar - tNear) / float(steps);

  // Opacity-correction exponent. Front-to-back compositing of N equal-alpha
  // samples accumulates total alpha '1 - (1-a)^N', which grows with N for a
  // fixed per-sample alpha 'a' — so reusing the same 'a' at e.g. 384 steps
  // ('full' quality) vs 128 steps ('interactive' quality, see
  // volumeRenderer.ts) would make the SAME volume look denser (and
  // eventually blow out toward solid color) purely from the step-count
  // change. To keep total accumulated opacity along a ray roughly constant
  // across step counts, treat uRawOpacity/label alpha as a "reference"
  // per-sample alpha calibrated at REF_STEPS samples, then solve for the
  // per-sample alpha at the ACTUAL step count that yields the same total:
  //   1 - (1-aRef)^REF_STEPS == 1 - (1-a)^steps
  //   => a = 1 - (1-aRef)^(REF_STEPS/steps)
  const float REF_STEPS = 256.0;
  float correctionExponent = REF_STEPS / float(steps);

  vec4 acc = vec4(0.0);
  for (int i = 0; i < steps; i++) {
    if (acc.a > 0.98) break; // early-exit once effectively opaque

    float t = tNear + (float(i) + 0.5) * stepSize;
    vec3 p = rayOrigin + rayDir * t;

    // World -> texture coords (see module-doc comment above for the Y-flip
    // rationale).
    vec3 tc = p / (2.0 * uBoxHalf) + 0.5;
    tc.y = 1.0 - tc.y;

    float rawValue = texture(uRawTex, tc).r;
    float v = clamp((rawValue - uWindow.x) / max(uWindow.y - uWindow.x, 1e-4), 0.0, 1.0);

    float rawAlphaRef = clamp(v * uRawOpacity, 0.0, 1.0);
    float rawAlpha = (uMode != 2) ? (1.0 - pow(1.0 - rawAlphaRef, correctionExponent)) : 0.0;

    int lbl = int(texture(uLabelTex, tc).r * 255.0 + 0.5);
    vec4 lc = texelFetch(uLutTex, ivec2(lbl, 0), 0);
    float labelAlphaRef = clamp(lc.a * uLabelOpacity, 0.0, 1.0);
    float labelAlpha = (uMode != 1) ? (1.0 - pow(1.0 - labelAlphaRef, correctionExponent)) : 0.0;

    vec3 rawColor = vec3(v);

    // "Realistic" mode: shade both the raw grayscale and the label color
    // using a Blinn-Phong model driven by the raw intensity gradient as an
    // approximate surface normal — the standard technique that turns a flat,
    // uniformly-lit volume raycast into something that reads as a lit solid
    // material instead of a translucent gel. Three things distinguish this
    // from a naive first attempt (see git history for the v1 headlamp-only
    // version, which read as nearly flat):
    //  1. The key light ('uLightDir') is offset from the camera axis — a
    //     light that travels WITH the view direction degenerates to a pure
    //     facing-ratio term (N.L ~= N.V), which barely varies across a
    //     rounded surface and reads as flat. 'uLightDir' is computed once per
    //     frame in volumeRenderer.ts from the camera's own basis (up-and-to
    //     the-side of the eye), so relief actually shows and shifts
    //     believably as the user orbits.
    //  2. A small headlamp FILL term ('nDotV') keeps silhouette-facing
    //     surfaces — facing the camera but turned away from the key light —
    //     from going pure black, the way a single off-axis light alone would.
    //  3. Shading is faded in by gradient magnitude ("surface-ness"): in a
    //     homogeneous region the gradient is pure numerical noise with a
    //     effectively random direction, so shading it produces a sparkling/
    //     shimmering artifact as the ray marches past voxel boundaries.
    //     'surfaceness' below suppresses this smoothly — flat interiors stay
    //     perfectly calm, only real intensity boundaries (grain surfaces)
    //     pick up any lighting at all.
    // Opt-in and off by default: 6 extra uRawTex fetches per step is a real
    // cost, only worth paying when the user asks for it.
    if (uShading == 1) {
      vec3 gradRaw = rawGradient(tc);
      float gradLen = length(gradRaw);
      if (gradLen > 1e-5) {
        // Outward normal: points from denser material toward less-dense
        // (matches the usual solid-object convention, where a normal points
        // away from the object's interior).
        vec3 normal = -gradRaw / gradLen;
        vec3 viewDir = normalize(uEye - p);
        vec3 halfVec = normalize(uLightDir + viewDir);

        float nDotL = max(dot(normal, uLightDir), 0.0);
        float nDotV = max(dot(normal, viewDir), 0.0);
        float nDotH = max(dot(normal, halfVec), 0.0);

        float ambient = 0.35;
        float diffuse = 0.55 * nDotL;
        float fill = 0.15 * nDotV;
        float specular = 0.4 * pow(nDotH, 32.0);
        float lit = clamp(ambient + diffuse + fill + specular, 0.0, 1.8);

        // Thresholds are eyes-on tunable, not derived from anything physical —
        // chosen so a typical normalized [0,1] intensity field's real grain
        // boundaries clear the top of the ramp while sensor/quantization noise
        // in flat regions stays below the bottom.
        float surfaceness = smoothstep(0.02, 0.15, gradLen);
        float lightAmount = mix(1.0, lit, surfaceness * uShadingStrength);

        rawColor *= lightAmount;
        lc.rgb *= lightAmount;
      }
    }

    // Composite this sample's raw contribution, then its label contribution,
    // as two sequential front-to-back "over" operations against the same
    // accumulator — equivalent to treating the label as a thin colored layer
    // drawn on top of the grayscale layer at this exact sample point. Either
    // term is a no-op (alpha 0) when its uMode branch above zeroed it out,
    // or when the LUT alpha is 0 for a hidden class.
    acc.rgb += (1.0 - acc.a) * rawColor * rawAlpha;
    acc.a += (1.0 - acc.a) * rawAlpha;

    acc.rgb += (1.0 - acc.a) * lc.rgb * labelAlpha;
    acc.a += (1.0 - acc.a) * labelAlpha;
  }

  // Accumulated front-to-back color/alpha. The canvas's own background
  // (its CSS/clear color, a caller concern, not this shader's) shows through
  // wherever alpha < 1.
  outColor = acc;
}
`;
