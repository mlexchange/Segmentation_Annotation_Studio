// WebGPU global types for the vendored volume renderer.
//
// `@webgpu/types` is not an `@types/*` package, so TypeScript does not pick it up
// automatically. Referencing it here rather than adding a `types` array to
// tsconfig.app.json is deliberate: setting `types` switches off the "include every
// @types package" default, which would silently drop the globals vitest and node
// contribute elsewhere in the app.
/// <reference types="@webgpu/types" />
