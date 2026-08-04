# Segmentation Annotation Studio — Code, Cybersecurity, and Structure Audit

**Audit date:** 2026-08-03  
**Scope:** Backend, frontend, Tiled integration, local launchers, Docker deployment, dependencies, tests, CI, documentation, and repository organization  
**Review type:** Source-assisted static review plus non-destructive build/test/dependency checks  
**Overall verdict:** A strong local/research prototype with useful domain separation and good algorithmic tests, but **not safe for networked or production use in its current form**.

## Executive summary

The project has a sensible top-level layout and a solid scientific-annotation foundation. Backend domain logic is split into focused modules; the frontend has pages, components, hooks, stores, and libraries; TypeScript is strict; Tiled access is routed through the backend; and the existing backend and frontend tests pass.

The security boundary, however, is not currently reliable. Both development launchers expose Vite on every interface, and Vite proxies `/api` to the loopback-only backend. The Docker configuration also publishes an unauthenticated backend that can read local files and perform privileged Tiled writes. In addition, arbitrary Tiled URIs can receive the default API key, the local-filesystem root is chosen by the caller, export paths are traversable, and Docker can copy `backend/.env` into image layers.

There are also several data-integrity bugs. A stale image can remain visible while edits are committed to a newly selected slice; transient draft or guide load failures can be followed by autosaves that overwrite existing work; multi-sample export parses ordinary Tiled keys incorrectly and can apply the wrong class taxonomy; and supposedly immutable versions can be overwritten by concurrent saves.

### Finding count

| Severity | Count | Meaning |
|---|---:|---|
| Critical | 4 | Stop-ship for any reachable deployment; plausible credential exposure, privileged remote actions, or broad compromise |
| High | 11 | Serious confidentiality, integrity, availability, or annotation-correctness risk |
| Medium | 17 | Important hardening, correctness, privacy, reproducibility, and maintainability work |

Severity assumes that another machine, browser origin, container peer, or untrusted local user can reach the application. If the tool is truly isolated to one trusted desktop user, some network findings have lower immediate likelihood, but the current launchers and Compose defaults break that assumption.

## Immediate actions

Before using this application with valuable or sensitive data:

1. Change both development launchers to `--host 127.0.0.1` and verify the listening sockets.
2. Change Compose to `127.0.0.1:8002:8002` until authentication and authorization exist.
3. Add `.env` patterns to `.dockerignore`; rotate the Tiled key if any image was built or pushed after `backend/.env` existed.
4. Reject arbitrary `server_uri` values and remove the default-key fallback for unconfigured servers.
5. Remove `server_api_key` from the public API and frontend completely.
6. Disable or guard Tiled write routes until server-enforced authentication, authorization, and explicit confirmation are implemented.
7. Restrict local files and exports to server-configured roots using opaque identifiers and verified containment.
8. Suspend `replace` ingest/mask workflows for valuable data until they validate and stage before deletion.
9. Fix slice/image identity handling and autosave load-error handling before production annotation work.

## Scope and methodology

The review covered:

- all Python, TypeScript, and TSX source files;
- FastAPI routes, schemas, local persistence, image processing, export/import, ingest, and Tiled client/write paths;
- React data flow, annotation state, image loading, autosave, version restore, export, browser integration, and accessibility;
- shell and PowerShell launchers, Dockerfile, Compose, Tiled configuration, and environment handling;
- Python and npm dependency declarations/locks;
- CI, tests, documentation, and high-level module structure.

No Tiled write endpoint was invoked. No destructive command was run. Temporary environments outside the repository were used for dependency installation and tests.

### Verification results

| Check | Result |
|---|---|
| Backend tests | **Pass:** 67 tests |
| Frontend tests | **Pass:** 16 files / 89 tests |
| Frontend TypeScript check | **Pass** |
| Frontend production build | **Pass** |
| Current flake8/isort check | **Pass** |
| Black 26.5.1 check | **Fail:** 30 files would be reformatted |
| npm audit, including and omitting dev dependencies | **9 advisories:** 6 high, 3 moderate |
| Frontend component tests | **None:** zero `.test.tsx` files |
| Browser/E2E tests | **None found** |

The frontend build also produced a roughly 23.6 MB WASM asset and a roughly 506 KB worker before compression. That is not a security flaw, but it should become a monitored performance budget.

---

## Critical findings

### C-01 — Development launcher exposes the privileged API to the LAN

**Evidence**

- Vite is correctly configured for loopback at [`frontend/vite.config.ts:48`](frontend/vite.config.ts#L48).
- Vite proxies `/api` to the FastAPI backend at [`frontend/vite.config.ts:51`](frontend/vite.config.ts#L51).
- The Unix launcher passes a bare `--host` at [`start_all.sh:616`](start_all.sh#L616).
- The Windows launcher does the same at [`windows/start_all.ps1:475`](windows/start_all.ps1#L475).
- The project promises loopback-only operation at [`README.md:9`](README.md#L9) and [`README.md:130`](README.md#L130).

Vite documents that `--host`/`host: true` listens on all addresses. The CLI argument overrides the safe `127.0.0.1` configuration. A machine on the same network can therefore contact Vite and send `/api/...` requests through its loopback proxy even though FastAPI itself listens only on loopback. See the official [Vite `server.host` documentation](https://vite.dev/config/server-options.html#server-host).

**Impact**

A LAN caller can reach local-file browsing, image rendering, draft/version data, export, ingest, and Tiled-write functionality. Because those API routes have no authentication, the Vite proxy becomes a remote entry point to the user's data and Tiled credential authority.

**Fix**

- Remove the CLI override or pass `--host 127.0.0.1` explicitly in both launchers.
- Add an automated launcher smoke test that inspects listening sockets and fails if any service binds a non-loopback address.
- Add `TrustedHostMiddleware` and a random per-launch session credential as defense in depth; loopback alone is not authorization.
- Document an explicit, separately secured network mode if remote access is desired.

**Regression test**

Start the stack and assert that frontend, backend, Tiled, and docs listeners are reachable on `127.0.0.1` but not on the host's LAN address.

### C-02 — Docker publishes an anonymous API with destructive Tiled authority

**Evidence**

- FastAPI has no application-level authentication dependency at [`backend/annotation_server.py:86`](backend/annotation_server.py#L86).
- Anonymous callers can save/sync annotations at [`backend/annotation_server.py:762`](backend/annotation_server.py#L762), write masks at [`backend/annotation_server.py:1117`](backend/annotation_server.py#L1117), and ingest/replace Tiled content at [`backend/annotation_server.py:1186`](backend/annotation_server.py#L1186).
- The container listens on `0.0.0.0` at [`Dockerfile:36`](Dockerfile#L36).
- Compose publishes `8002:8002` at [`docker-compose.yml:9`](docker-compose.yml#L9), which publishes on all host addresses by default. See Docker's [port-publishing documentation](https://docs.docker.com/engine/network/port-publishing/).
- The generated OpenAPI schema contains no security schemes or operation-level security requirements.

**Impact**

Any reachable client can use the backend as a confused deputy: enumerate/render readable host data, inspect annotation history, spoof attribution, create exports, exhaust resources, and use the backend-held Tiled key to modify or replace Tiled content. CORS does not authenticate requests and does not protect non-browser clients.

**Fix**

- For local mode, publish `127.0.0.1:8002:8002`, validate `Host` and `Origin`, and issue a random per-launch session credential.
- For network mode, require authenticated identities behind TLS, then authorize distinct read, annotate, export, ingest, replace, and mask-write scopes.
- Enforce CSRF protection for cookie-authenticated writes.
- Require a server-issued, short-lived confirmation token bound to the target, operation, and expected revision for destructive actions.
- Add audit logging for every privileged operation without logging secrets or raw sensitive payloads.

### C-03 — Arbitrary Tiled URI creates SSRF and default-key exfiltration risk

**Evidence**

- Client-controlled `server_uri` is accepted throughout browse, image, export, mask, and ingest flows; representative routes start at [`backend/annotation_server.py:173`](backend/annotation_server.py#L173), [`backend/annotation_server.py:248`](backend/annotation_server.py#L248), [`backend/annotation_server.py:489`](backend/annotation_server.py#L489), and [`backend/annotation_server.py:1186`](backend/annotation_server.py#L1186).
- [`backend/tiled_clients.py:33`](backend/tiled_clients.py#L33) uses the supplied URI without an allowlist.
- [`backend/tiled_clients.py:34`](backend/tiled_clients.py#L34) selects `server_api_key or api_key_for_uri(uri) or get_tiled_api_key()`.
- [`backend/tiled_clients.py:44`](backend/tiled_clients.py#L44) passes that key to `tiled.client.from_uri()`.

For an unconfigured URI, the default Tiled key is selected. An attacker-controlled server that behaves enough like Tiled can receive the credential on an authenticated follow-up request. Other internal targets can still be probed through server-side requests. The process-wide `TILED_API_KEY` environment variable may also be consumed implicitly by the Tiled client, so merely passing `None` is not a complete defense.

**Impact**

- Exfiltration of the key that authorizes Tiled writes.
- SSRF against internal services, container peers, and cloud metadata endpoints.
- Unbounded client-cache growth using attacker-selected URIs.

**Fix**

- Replace public URIs with opaque `server_id` values mapped to exact server-side configuration.
- Reject every unconfigured server; canonicalize scheme/host/port and control redirects.
- Assign one credential to one configured server and never fall back to a global key for a foreign URI.
- Remove the process-wide credential from the environment seen by generic clients where feasible.
- Require HTTPS except for explicitly configured loopback development servers.
- Apply network egress restrictions, short timeouts, response-size limits, and a bounded client cache.

This follows OWASP's recommended allowlist-oriented defense for SSRF: [SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html).

**Regression test**

Use a local capture server and prove that an unknown `server_id` is rejected before any connection and that the default key is never attached to an unconfigured destination.

### C-04 — Docker can copy `backend/.env` and the Tiled key into image layers

**Evidence**

- [`.dockerignore`](.dockerignore) does not exclude `.env` files.
- [`start_all.sh:381`](start_all.sh#L381) creates/populates `backend/.env` and later writes the generated key.
- [`Dockerfile:28`](Dockerfile#L28) runs `COPY backend/ ./` in the final stage.

**Impact**

After the local launcher creates `backend/.env`, a subsequent image build can include it in a layer and in the final filesystem. Anyone with access to the image, registry, layer cache, exported tar, or some CI artifacts may recover the Tiled credential. Deleting the file in a later Docker layer would not remove it from earlier layers.

**Fix**

- Add `**/.env` and `**/.env.*` to `.dockerignore`, followed by explicit exceptions for `!**/.env.example`.
- Inject runtime secrets through an orchestrator secret mechanism or protected runtime environment, never with `COPY`, `ARG`, or build logs.
- Add a build-context secret scan in CI.
- Rotate the Tiled key and rebuild/purge images if any image was built or distributed after `.env` existed.

---

## High findings

### H-01 — The local-filesystem sandbox root is selected by the caller

**Evidence**

- [`backend/local_fs.py:41`](backend/local_fs.py#L41) resolves any client-provided root.
- [`backend/local_fs.py:79`](backend/local_fs.py#L79) accepts any absolute source path when no root is supplied.
- Public routes expose the root/source at [`backend/annotation_server.py:397`](backend/annotation_server.py#L397), [`backend/annotation_server.py:406`](backend/annotation_server.py#L406), [`backend/annotation_server.py:549`](backend/annotation_server.py#L549), and [`backend/annotation_server.py:581`](backend/annotation_server.py#L581).
- Export resolution also reaches local arrays without a fixed server-owned root at [`backend/annotation_server.py:961`](backend/annotation_server.py#L961).

The existing resolved-path containment logic is reasonable when the root is trusted. The flaw is that the request chooses that root or bypasses it with an absolute path.

**Impact**

An API caller can enumerate `/` or another readable directory and obtain rendered versions of supported images, TIFFs, or NumPy arrays anywhere the process can read. COCO import likewise accepts an arbitrary server directory at [`backend/annotation_server.py:1149`](backend/annotation_server.py#L1149).

**Fix**

- Configure allowed roots server-side and expose only opaque root/source IDs.
- Reject absolute client paths and any source not registered beneath an allowed root.
- Re-check containment after resolving symlinks with `Path.is_relative_to()`.
- Centralize source authorization so image, measure, guide, export, draft, and thumbnail paths cannot bypass it.

### H-02 — Export path traversal and unsafe ZIP names allow writes outside the export root

**Evidence**

- `dataset_name` becomes a path component at [`backend/annotation_server.py:873`](backend/annotation_server.py#L873).
- The containment check at [`backend/annotation_server.py:888`](backend/annotation_server.py#L888) uses string `startswith`, so a sibling such as `exports_evil` can pass for an `exports` root.
- `split_by_slice` values are unconstrained strings in [`backend/schemas.py:214`](backend/schemas.py#L214).
- Split values flow to filesystem and archive prefixes at [`backend/annotation_server.py:1045`](backend/annotation_server.py#L1045), [`backend/annotation_server.py:1054`](backend/annotation_server.py#L1054), and [`backend/coco_export.py:277`](backend/coco_export.py#L277).
- An absolute child passed to `out_root / split_name` discards `out_root`; `..` values can traverse as well.
- `mode="fail"` writes the ZIP/manifest before checking one existing output at [`backend/annotation_server.py:1031`](backend/annotation_server.py#L1031) and [`backend/coco_export.py:280`](backend/coco_export.py#L280).

**Impact**

A caller can cause server-side files such as COCO JSON, PNG masks, or archives to be created or overwritten outside the intended export directory. Unsafe archive members can also create a downstream ZIP-slip risk for consumers.

**Fix**

- Constrain split values to `Literal["train", "valid", "test", "auto"]`.
- Validate `dataset_name` as one safe filename component; reject separators, absolute paths, `.` and `..`.
- Resolve every output and require `target.is_relative_to(resolved_export_root)`.
- Validate POSIX archive member names separately.
- Perform all conflict checks first, write into a private staging directory, validate, then publish atomically.

### H-03 — Destructive Tiled replacement deletes before successful validation/publication

**Evidence**

- Ingest deletes the existing node at [`backend/ingest.py:362`](backend/ingest.py#L362) before reading/validating the replacement at [`backend/ingest.py:363`](backend/ingest.py#L363).
- Mask sync treats broad read failures as if no prior mask exists at [`backend/tiled_mask_sync.py:185`](backend/tiled_mask_sync.py#L185).
- It deletes prior contents at [`backend/tiled_mask_sync.py:265`](backend/tiled_mask_sync.py#L265) and then performs multiple writes beginning at [`backend/tiled_mask_sync.py:273`](backend/tiled_mask_sync.py#L273).
- Direct API calls can request replacement without proving the UI confirmation occurred.

**Impact**

A corrupt upload, transient network failure, class-name collision, process crash, or concurrent job can erase valid content and leave nothing or a partial mask tree. A temporary read error can incorrectly switch behavior from merge to replacement.

**Fix**

- Decode, validate, size-check, and stage the full replacement before mutating the target.
- Publish a new version/key, verify it, then atomically switch an alias/pointer if supported.
- Preserve a recoverable prior version and add rollback.
- Bind a one-time confirmation token to the target and its expected current revision.
- Serialize or compare-and-swap writes to the same target.
- Never interpret an unexpected read failure as “target absent.”

### H-04 — Unbounded uploads, geometry, threads, and job registries enable denial of service

**Evidence**

- Multipart upload at [`backend/annotation_server.py:1186`](backend/annotation_server.py#L1186) has no body, per-file, aggregate, count, decoded-pixel, or array-shape limit.
- Image/array decoders materialize uploaded data at [`backend/ingest.py:112`](backend/ingest.py#L112).
- Successful ingest removes individual files but not the parent temporary directory at [`backend/ingest.py:400`](backend/ingest.py#L400); earlier failures do not consistently clean up.
- Every export/job can create daemon threads; export creates up to 16 more workers at [`backend/coco_export.py:596`](backend/coco_export.py#L596).
- Job/client dictionaries never expire at [`backend/export_jobs.py:14`](backend/export_jobs.py#L14), [`backend/ingest.py:58`](backend/ingest.py#L58), and [`backend/tiled_clients.py:15`](backend/tiled_clients.py#L15).
- Raw brush geometry can trigger extremely long coordinate loops at [`backend/coco_export.py:75`](backend/coco_export.py#L75).
- Browse thumbnails read the full array before downsampling at [`backend/thumbnails.py:40`](backend/thumbnails.py#L40).

**Impact**

An unauthenticated caller can exhaust disk, RAM, CPU, threads, network connections, or browser memory. Image decompression bombs and huge array headers/shapes are particularly important for scientific file formats.

**Fix**

- Enforce request-body limits before multipart parsing and file/count/aggregate quotas while streaming.
- Inspect headers safely, then enforce decoded dimensions, pixels, dtype, rank, and total bytes before allocation.
- Bound annotation/class/slice counts and all numeric coordinates; reject non-finite values.
- Use a bounded queue with per-principal concurrency/rate limits, cancellation, TTL eviction, and durable status.
- Clean temporary directories in `finally` blocks and on startup.
- Downsample at the data source rather than materializing full arrays for thumbnails.

### H-05 — Public API and frontend still accept Tiled keys in query strings

**Evidence**

- `server_api_key` is public on four routes at [`backend/annotation_server.py:176`](backend/annotation_server.py#L176), [`backend/annotation_server.py:251`](backend/annotation_server.py#L251), [`backend/annotation_server.py:294`](backend/annotation_server.py#L294), and [`backend/annotation_server.py:335`](backend/annotation_server.py#L335).
- Frontend `ServerInfo` includes `api_key` at [`frontend/src/types/server.ts:1`](frontend/src/types/server.ts#L1).
- [`frontend/src/components/Browse/hooks/useBrowseData.ts:92`](frontend/src/components/Browse/hooks/useBrowseData.ts#L92) accepts it and adds `server_api_key` to GET URLs.
- Browse cache keys omit the credential/security context at [`backend/annotation_server.py:262`](backend/annotation_server.py#L262), [`backend/annotation_server.py:304`](backend/annotation_server.py#L304), and [`backend/annotation_server.py:343`](backend/annotation_server.py#L343).

**Impact**

Credentials can leak through URLs, browser history, access/proxy logs, diagnostics, screenshots, and referrers. A privileged response can also be cached under the same key later used by a request without that privilege.

**Fix**

- Remove the parameter from schemas, OpenAPI, backend functions, TypeScript types, and request builders.
- Resolve credentials only from the exact server-side server record.
- Partition any protected cache by authenticated principal/tenant and credential scope, or do not share it.
- Add contract tests that API responses, requests, logs, and OpenAPI contain no key field/parameter.

### H-06 — Concurrent persistence can overwrite drafts and immutable versions

**Evidence**

- Version allocation uses `len(existing) + 1` without exclusive creation at [`backend/drafts.py:156`](backend/drafts.py#L156). Concurrent saves select the same number, and gaps can reuse an existing version.
- Draft and guide writes share fixed `.tmp` names at [`backend/drafts.py:70`](backend/drafts.py#L70) and [`backend/guides.py:54`](backend/guides.py#L54).
- Client PUTs replace full documents without a revision/ETag.
- Concurrent exports to one dataset and concurrent mask merges have equivalent lost-update/partial-output risks.

**Impact**

Multiple tabs, fast saves, multiple workers, or overlapping jobs can lose newer work, corrupt a temporary file, overwrite a numbered historical version, or publish mixed outputs.

**Fix**

- Move persisted metadata to a transactional store such as SQLite.
- Add unique `(source_key, version)` constraints and allocate with a transaction or `max(version)+1` under a lock.
- Add document revisions/ETags and return `409 Conflict` on stale writes.
- Use unique temporary files and atomic replace only after fsync/validation.
- Serialize conflicting export/mask operations by target; process-local locks alone are insufficient for multiple workers.

### H-07 — Slice changes can display old pixels while committing edits to the new slice

**Evidence**

- Slice data changes with the current selection at [`frontend/src/components/annotate/AnnotationCanvas/index.tsx:519`](frontend/src/components/annotate/AnnotationCanvas/index.tsx#L519).
- `imageEl` is not cleared on a new request; there is no `onerror` or request-generation check at [`frontend/src/components/annotate/AnnotationCanvas/index.tsx:520`](frontend/src/components/annotate/AnnotationCanvas/index.tsx#L520).
- Commits read the current store slice, not the identity of the rendered image, at [`frontend/src/components/annotate/AnnotationCanvas/index.tsx:760`](frontend/src/components/annotate/AnnotationCanvas/index.tsx#L760).
- SAM and classic-wand caches label `imageEl` with the current source/slice at [`frontend/src/components/annotate/AnnotationCanvas/index.tsx:558`](frontend/src/components/annotate/AnnotationCanvas/index.tsx#L558) and [`frontend/src/components/annotate/AnnotationCanvas/index.tsx:816`](frontend/src/components/annotate/AnnotationCanvas/index.tsx#L816).

**Impact**

A slow, failed, or reordered image load can leave a previous slice visible and editable indefinitely while shapes are saved to the newly selected slice. Segmentation caches can also be computed from old pixels under a new key. This is silent scientific annotation corruption.

**Fix**

- Store loaded state as `{requestKey, image}` and clear/disable the canvas immediately when the key changes.
- Abort superseded fetches, reject stale image callbacks, and show loading/error overlays.
- Permit edit/segmentation commits only when rendered identity exactly matches the active source and slice.
- Key derived caches to the successfully loaded generation, not mutable current state.
- Add delayed, reordered, failed, and rapid-navigation tests.

### H-08 — Draft and guide load failures can erase existing work

**Evidence**

- [`frontend/src/hooks/useDraftSync.ts:95`](frontend/src/hooks/useDraftSync.ts#L95) maps every non-404/network failure to `null`, indistinguishable from no draft.
- Opening proceeds with empty/default state at [`frontend/src/hooks/useOpenInAnnotate.ts:40`](frontend/src/hooks/useOpenInAnnotate.ts#L40).
- Autosave PUTs current state after 1.5 seconds even without a user edit at [`frontend/src/hooks/useDraftSync.ts:50`](frontend/src/hooks/useDraftSync.ts#L50).
- `putDraft` never checks `res.ok` at [`frontend/src/hooks/useDraftSync.ts:13`](frontend/src/hooks/useDraftSync.ts#L13).
- The unload fallback uses `sendBeacon` (POST) at [`frontend/src/hooks/useDraftSync.ts:80`](frontend/src/hooks/useDraftSync.ts#L80), but the backend exposes only PUT at [`backend/annotation_server.py:637`](backend/annotation_server.py#L637).
- Guide loading has the same error-to-null behavior at [`frontend/src/hooks/useGuideSync.ts:17`](frontend/src/hooks/useGuideSync.ts#L17), then installs/auto-saves an empty guide beginning at [`frontend/src/hooks/useGuideSync.ts:87`](frontend/src/hooks/useGuideSync.ts#L87).

**Impact**

A transient GET error followed by a successful PUT can overwrite an existing draft or guide with empty/current defaults. Guide cleanup can also discard a pending last edit.

**Fix**

- Represent `loading`, `found`, `not-found`, and `error` distinctly.
- Enable autosave only after a successful load or a confirmed 404, and only after a dirty edit.
- Surface retry/error state and preserve a local recovery copy, preferably IndexedDB.
- Check response status, serialize saves, and use revision/ETag conflict control.
- Either implement a deliberately authenticated compatible beacon endpoint or use a tested `keepalive` strategy; do not claim the current POST beacon flush succeeds.

### H-09 — Multi-sample export breaks ordinary Tiled sources and can mislabel classes

**Evidence**

- Tiled keys are built as `tiled:${serverUri}:${path}` at [`frontend/src/lib/sourceKey.ts:13`](frontend/src/lib/sourceKey.ts#L13).
- Export parses at the first colon at [`frontend/src/components/annotate/DownloadModal.tsx:20`](frontend/src/components/annotate/DownloadModal.tsx#L20). A key containing `http://...` produces `serverUri === "http"`.
- Classes are loaded as sample-specific state at [`frontend/src/hooks/useOpenInAnnotate.ts:95`](frontend/src/hooks/useOpenInAnnotate.ts#L95).
- “All samples” sends every annotation map with only the currently open global class list at [`frontend/src/components/annotate/DownloadModal.tsx:68`](frontend/src/components/annotate/DownloadModal.tsx#L68) and [`frontend/src/components/annotate/DownloadModal.tsx:121`](frontend/src/components/annotate/DownloadModal.tsx#L121).

**Impact**

Multi-source export can fail to reconnect to Tiled or silently map class IDs using the wrong taxonomy, producing scientifically incorrect training labels.

**Fix**

- Replace delimiter parsing with a versioned structured identifier and one canonical encode/decode implementation.
- Persist/load a taxonomy per source or explicitly enforce one shared taxonomy and validate compatibility before merge.
- Add round-trip tests for HTTP(S), ports, IPv6, Unicode paths, local paths, and migrated historic keys.
- Validate exported class references against the chosen taxonomy before writing any output.

### H-10 — API schemas bypass the documented annotation model

**Evidence**

- Polygon/rectangle/ellipse/brush models exist, but draft/export/measure payloads use `dict[str, Any]` at [`backend/schemas.py:169`](backend/schemas.py#L169), [`backend/schemas.py:232`](backend/schemas.py#L232), and [`backend/schemas.py:262`](backend/schemas.py#L262).
- Rectangle/ellipse normalization, finite coordinates, point parity/counts, positive radii, class uniqueness/references, string lengths, collection sizes, split values, and auto-split ratios are not enforced at the API boundary.
- Sanitized class names can collide in mask keys; separate user names such as punctuation variants can map to the same safe name.

**Impact**

Malformed or huge data can violate the core coordinate model, corrupt exports/masks, reference missing classes, generate key collisions, or amplify DoS paths. NaN/Infinity values can propagate unpredictably through geometry and JSON/image operations.

**Fix**

- Use a discriminated `Shape` union at every boundary.
- Add constrained finite numeric types, normalized dimensions/radii, bounded point/stroke/shape/class/slice counts, and `extra="forbid"`.
- Validate unique class IDs/names, shape-to-class references, colors, split enums/ratios, and format class-count limits.
- Make collision-resistant storage keys include a stable ID/hash, not only a sanitized label.
- Generate TypeScript contracts from the authoritative schema and keep persisted migrations versioned.

### H-11 — Undo, version restore, and dirty state do not describe one coherent editor state

**Evidence**

- Temporal snapshots include only `byImage` and the in-progress draft at [`frontend/src/stores/annotationStore.ts:470`](frontend/src/stores/annotationStore.ts#L470).
- Split and negative state are changed outside that history at [`frontend/src/stores/annotationStore.ts:399`](frontend/src/stores/annotationStore.ts#L399).
- Version restore replaces global classes plus all source annotation fields at [`frontend/src/hooks/useSave.ts:206`](frontend/src/hooks/useSave.ts#L206), but undo can restore only shapes/draft.
- Two initialization effects around [`frontend/src/hooks/useSave.ts:87`](frontend/src/hooks/useSave.ts#L87) and [`frontend/src/hooks/useSave.ts:96`](frontend/src/hooks/useSave.ts#L96) can cause the first real edit after mount to be treated as initialization; the UI can continue to report “Saved.”

**Impact**

Undo/redo can leave classes, shapes, split assignment, and negative flags from different logical revisions. Users can also navigate away believing a first edit was saved.

**Fix**

- Define one source-scoped editor snapshot containing taxonomy, shapes, splits, negative flags, and in-progress tool state.
- Treat version restore and class operations as compound temporal transactions.
- Track dirty state against a successfully persisted revision/fingerprint rather than effect ordering.
- Add tests covering undo/redo around class deletion, restore, split/negative changes, first edit, autosave, and source switching.

---

## Medium findings

### M-01 — Dependency advisories and duplicated/unused frontend dependency trees

A current audit of the committed lockfile reported **9 affected dependency nodes: 6 high and 3 moderate**. Locked versions include:

| Package | Locked version | Context |
|---|---:|---|
| `react-router` | 7.17.0 | Some advisories concern Framework/RSC modes not used by this client-rendered SPA; update nevertheless |
| `react-router-dom` | 6.30.4 | Older transitive tree pulled by Finch/Tiled dependencies |
| `postcss` | 8.5.15 | Build-time exposure; update to a patched release |
| `uuid` | 10.0.0 | Deprecated/unsupported lock; reported affected APIs are not the app's observed v4 path |
| `protobufjs` | 7.6.4 | Transitive parser/runtime exposure |
| `@huggingface/transformers` | 4.2.0 | Pulls the ONNX/Sharp/adm-zip chain |
| `onnxruntime-node` | 1.24.3 | Node-side transitive dependency, likely install/build rather than browser runtime |
| `sharp` | 0.34.5 | Node image-processing advisory; contextual reachability appears indirect here |
| `adm-zip` | 0.5.17 | Crafted ZIP resource-exhaustion advisory; indirect Node path |

Relevant advisory examples include [PostCSS GHSA-r28c-9q8g-f849](https://github.com/advisories/GHSA-r28c-9q8g-f849), [React Router GHSA-chx6-hx7r-mcp5](https://github.com/advisories/GHSA-chx6-hx7r-mcp5), [React Router RSC GHSA-qwww-vcr4-c8h2](https://github.com/advisories/GHSA-qwww-vcr4-c8h2), [UUID GHSA-w5hq-g745-h8pq](https://github.com/advisories/GHSA-w5hq-g745-h8pq), and [adm-zip GHSA-xcpc-8h2w-3j85](https://github.com/advisories/GHSA-xcpc-8h2w-3j85).

The numerical “6 high” result should not be misread as six directly exploitable browser vulnerabilities. Several findings are mode-specific or transitive Node/build paths. They still create maintenance and supply-chain risk and should be resolved or documented with time-limited reachability waivers.

Additional evidence:

- [`frontend/package.json:15`](frontend/package.json#L15) declares `@blueskyproject/finch`, but no source import was found.
- That dependency brings Storybook, Three, Tiled, and a second React Router generation into the production dependency graph.
- Documentation advertises Node 18 at [`README.md:13`](README.md#L13), while the locked router requires Node 20+, CI uses Node 20, and [`Dockerfile:7`](Dockerfile#L7) uses Node 22.

**Fix**

- Update direct dependencies and regenerate/test the lockfile.
- Remove unused Finch/Tiled frontend dependencies or deliberately integrate only the required shell package.
- Track upstream fixes/overrides for Transformers' optional Node chain and verify that pruning optional Node dependencies does not break browser inference.
- Standardize on at least Node 20; add `engines`, `packageManager`, and `.nvmrc`/Volta configuration.
- Add Dependabot/Renovate, an SBOM, and CI dependency scanning with documented, expiring waivers based on reachability.

### M-02 — SlimSAM model acquisition is mutable and nondeterministic

**Evidence**

- Launchers download an unpinned Hugging Face snapshot at [`start_all.sh:452`](start_all.sh#L452) and [`windows/start_all.ps1:303`](windows/start_all.ps1#L303).
- [`frontend/scripts/fetch-sam-model.mjs:44`](frontend/scripts/fetch-sam-model.mjs#L44) also omits an immutable revision.
- Runtime fallback fetches the remote model in [`frontend/src/lib/sam/samWorker.ts:35`](frontend/src/lib/sam/samWorker.ts#L35).
- `frontend/public/models` is Git-ignored but not Docker-ignored, so [`Dockerfile:11`](Dockerfile#L11) silently bakes a local copy into some images while other images depend on a remote fallback.
- The launcher treats any non-empty model directory as complete, including a partial download.

**Impact**

Two builds from the same source can use different model artifacts. A compromised upstream account/CDN, partial download, or silent model update can change annotation behavior or introduce malicious model/parser input without a source change.

**Fix**

- Choose explicitly whether production images include the model or require an offline-provisioned asset.
- Pin an immutable repository commit/revision and verify a checked-in manifest of cryptographic hashes.
- Download to a temporary directory, verify every required file, then rename atomically and write a completion marker.
- Pin `huggingface_hub` and installer tooling.
- Disable silent remote fallback in hardened/offline builds and expose model provenance in diagnostics.

### M-03 — Python environments and bootstrap tooling are not reproducible

**Evidence**

- [`backend/pyproject.toml:9`](backend/pyproject.toml#L9) uses broad lower bounds and there is no Python lock/hashed constraints file.
- The launchers maintain independent dependency lists at [`start_all.sh:277`](start_all.sh#L277) and [`windows/start_all.ps1:197`](windows/start_all.ps1#L197) rather than installing/synchronizing the project.
- They only install after selected imports fail, so an existing environment stays stale after dependency changes.
- The Unix list omits `python-multipart`; Pydantic and optional PyYAML are directly imported but not direct project requirements.
- [`start_all.sh:247`](start_all.sh#L247) and the Windows equivalent execute a remote installer script without an artifact checksum.

**Impact**

Local, CI, and Docker environments can resolve different package versions and behave differently. A future upstream release can break a fresh install without a repository change. Executing mutable remote installers expands supply-chain exposure.

**Fix**

- Commit `uv.lock` or hashed constraints and run `uv sync --locked` everywhere.
- Define explicit local-server, test, lint, and docs dependency groups; declare direct imports directly.
- Remove handwritten dependency installation logic from launchers.
- Pin or vendor installer artifacts with checksum/signature verification.
- Add an automated lock refresh workflow that runs the full test/build suite.

### M-04 — Raw errors, filesystem paths, and sensitive images are exposed to clients/caches

**Evidence**

- Many routes include raw exception strings in HTTP responses; examples include [`backend/annotation_server.py:285`](backend/annotation_server.py#L285), [`backend/annotation_server.py:546`](backend/annotation_server.py#L546), [`backend/annotation_server.py:578`](backend/annotation_server.py#L578), [`backend/annotation_server.py:1096`](backend/annotation_server.py#L1096), and [`backend/annotation_server.py:1162`](backend/annotation_server.py#L1162).
- Draft/save/export job responses can expose absolute server paths.
- Image and thumbnail routes use `Cache-Control: public` at [`backend/annotation_server.py:393`](backend/annotation_server.py#L393), [`backend/annotation_server.py:624`](backend/annotation_server.py#L624), and [`backend/annotation_server.py:832`](backend/annotation_server.py#L832).

**Impact**

Error bodies disclose server layout, data names, upstream details, and possibly credential-bearing URLs from third-party exceptions. Shared browser/proxy caches may retain sensitive scientific imagery after authorization context changes.

**Fix**

- Return stable public error codes/messages and a correlation ID; log full details server-side after secret/URL sanitization.
- Return opaque download/job IDs instead of absolute paths.
- Use `private, no-store` for sensitive images unless an authenticated cache design with correct `Vary`/key partitioning is implemented.
- Add tests asserting error redaction and cache headers.

### M-05 — Feedback silently sends dataset and browser context to Google

**Evidence**

- The application claims “no data leaves your machine” at [`README.md:9`](README.md#L9).
- A Google Form URL and entry ID are enabled by default at [`frontend/src/config.ts:18`](frontend/src/config.ts#L18).
- Feedback context contains the source/path, dimensions, dtype, browser/user agent, route/tool, GPU state, and persistent session ID at [`frontend/src/lib/feedbackContext.ts:22`](frontend/src/lib/feedbackContext.ts#L22).
- Opening the feedback modal constructs an iframe URL containing that context at [`frontend/src/app/App.tsx:103`](frontend/src/app/App.tsx#L103). The external iframe request occurs before form submission.

**Impact**

Local filesystem paths can contain usernames, study identifiers, sample names, or customer/project information. This contradicts the privacy promise and may violate organizational data-handling requirements.

**Fix**

- Disable external feedback by default; require deployment opt-in.
- Show exactly what will be sent and require explicit consent before creating the remote request.
- Remove/hash dataset paths and persistent session IDs; minimize browser/device details.
- Update privacy documentation to distinguish local image processing from optional external feedback/model traffic.

### M-06 — External frames and guide images are insufficiently constrained

**Evidence**

- [`frontend/src/components/IframeModal.tsx:46`](frontend/src/components/IframeModal.tsx#L46) embeds configured URLs without a restrictive `sandbox`, `referrerPolicy`, or origin allowlist.
- Production docs default to the browser user's `http://127.0.0.1:8000` at [`frontend/src/config.ts:16`](frontend/src/config.ts#L16).
- Imported guide `exampleCrops` are treated as arbitrary strings at [`frontend/src/app/pages/ReferencePage.tsx:183`](frontend/src/app/pages/ReferencePage.tsx#L183) and rendered as `<img src>` in Reference/Class Manager.
- Guide JSON/images have no byte, count, or decoded-dimension limits.
- No Content Security Policy or complete browser security-header policy was found.

**Impact**

Untrusted guide content can trigger browser requests to arbitrary external or intranet locations. An arbitrary deployment-configured iframe receives more capabilities/referrer information than necessary. The default docs URL can unintentionally target a service on the end user's own machine.

**Fix**

- Allowlist HTTPS origins/schemes and use a least-privilege iframe sandbox/`allow` policy plus `referrerPolicy="no-referrer"` where compatible.
- Disable the production Docs button unless a deliberate URL is configured.
- Accept only validated, size-limited `data:image/png;base64` guide crops and decode/re-encode them before storage.
- Add CSP, `frame-ancestors`, `X-Content-Type-Options`, a referrer policy, and other appropriate headers, accounting for required WASM/worker sources.

### M-07 — Object URLs and row-level subscriptions can exhaust browser memory

**Evidence**

- [`frontend/src/hooks/useImageSlice.ts:50`](frontend/src/hooks/useImageSlice.ts#L50) creates a new blob URL without revoking it.
- Item/slice rows independently call annotation-status hooks and subscribe to broad query/store data; representative sites are [`frontend/src/components/Browse/ItemsColumn.tsx:151`](frontend/src/components/Browse/ItemsColumn.tsx#L151) and [`frontend/src/components/Browse/SlicesColumn.tsx:128`](frontend/src/components/Browse/SlicesColumn.tsx#L128).
- Long item, slice, and facet lists are not virtualized.

**Impact**

Navigating through large stacks permanently retains decoded response blobs for the page lifetime. Thousands of row subscriptions and DOM nodes can make realistic scientific datasets sluggish or crash the tab.

**Fix**

- Revoke object URLs on query eviction/replacement/unmount and pass TanStack Query's `AbortSignal` to `fetch`.
- Compute annotation status once in the parent and pass narrow primitive props.
- Virtualize large lists and profile representative high-volume datasets.
- Add a memory/navigation stress test and performance budgets.

### M-08 — Browse/filter errors can fail open or show stale results

**Evidence**

- Invalid filter JSON becomes an empty filter at [`backend/annotation_server.py:1250`](backend/annotation_server.py#L1250).
- Tiled filter application errors are silently skipped at [`backend/browse_helpers.py:476`](backend/browse_helpers.py#L476), potentially broadening results.
- Frontend item loads always commit their response, while rapid selections can start overlapping requests without cancellation/generation protection at [`frontend/src/components/Browse/hooks/useBrowseData.ts:195`](frontend/src/components/Browse/hooks/useBrowseData.ts#L195) and [`frontend/src/components/Browse/hooks/useBrowseData.ts:305`](frontend/src/components/Browse/hooks/useBrowseData.ts#L305).

**Impact**

Users can see a broader or older dataset than the selected restrictions indicate, which is both a privacy/authorization problem and a scientific workflow correctness issue.

**Fix**

- Return `422` for invalid filter syntax and a clear upstream error if a requested restriction cannot be enforced.
- Never silently remove an access/data restriction.
- Key loads by the complete filter state and use TanStack Query, or abort superseded requests and compare a request generation before committing.
- Add out-of-order response and invalid-filter tests.

### M-09 — Scientific rendering and stack resolution can silently produce wrong results

**Evidence**

- Global statistics are computed in the raw domain at [`backend/images.py:62`](backend/images.py#L62), while log/symlog transforms occur before applying that range at [`backend/images.py:102`](backend/images.py#L102).
- Requested percentiles are not fully constrained.
- Container resolution follows the first child without proving the wrapper is unambiguous at [`backend/arrays.py:100`](backend/arrays.py#L100).
- Stack construction assumes later slice shapes match the first at [`backend/arrays.py:235`](backend/arrays.py#L235).
- Invalid stack indices can map to slice zero at [`backend/arrays.py:297`](backend/arrays.py#L297), inconsistent with normal NumPy indexing/error behavior.

**Impact**

Log/symlog display normalization can be numerically wrong, the wrong child array can be selected, mismatched stacks can fail late, and an invalid index can display a valid but unintended slice. These are silent scientific correctness risks.

**Fix**

- Compute normalization bounds in the transformed domain and validate percentile ordering/ranges.
- Require unambiguous container structures or an explicit child selector.
- Validate every slice's shape/dtype before building a stack.
- Define and test consistent bounds/negative-index behavior.
- Add numerical-reference tests across linear, log, symlog, constant, NaN, and extreme-value arrays.

### M-10 — Export/import completeness and identity guarantees are weak

**Evidence**

- [`backend/coco_import.py:1`](backend/coco_import.py#L1) promises a lossless `_studio_shapes.json` sidecar, but no export path writes it.
- Multi-source filenames use only a sanitized suffix of the source at [`backend/coco_export.py:516`](backend/coco_export.py#L516), allowing collisions for similar endings.
- Metadata/thumbnail synchronization errors are suppressed while save still reports success at [`backend/annotation_server.py:810`](backend/annotation_server.py#L810).
- `/api/import/coco` has no frontend path or round-trip tests and reads an arbitrary host directory.

**Impact**

Export/import may not preserve native shape semantics, multiple sources can overwrite or duplicate output names, and the UI can report a complete save while Tiled metadata remains stale.

**Fix**

- Implement and validate a versioned native sidecar, or remove the lossless claim.
- Include a stable source ID/hash in filenames and reject collisions before writing.
- Return separate local-save and Tiled-sync state; use a durable retry/outbox for eventual synchronization.
- Restrict import to configured roots or explicit uploads and add full export/import fixture tests.

### M-11 — Persisted documents are unversioned and stores are not cleanly separated

**Evidence**

- Draft JSON is persisted without a `schema_version` at [`backend/drafts.py:64`](backend/drafts.py#L64).
- Persisted Zustand stores have no explicit version/migration handlers.
- Draft listing globs every JSON document at [`backend/drafts.py:106`](backend/drafts.py#L106), while guides use `*.guide.json` at [`backend/guides.py:35`](backend/guides.py#L35); a guide can therefore be listed as a draft.
- Process-local job records disappear on restart/multi-worker routing and completed records are not evicted.

**Impact**

Future model changes can strand or reinterpret existing annotations. Document types leak into one another, and status polling becomes inconsistent across restarts/workers.

**Fix**

- Add explicit document/store schema versions and fixture-based migration tests.
- Separate draft, guide, version, and job storage by schema/table/directory and validate a document type discriminator.
- Use durable job state for production; at minimum add TTL/max eviction and keep a single worker until state is shared.

### M-12 — Several frontend controls are functionally broken or misleading

**Evidence**

- Tab customization merges every omitted route back into the selection at [`frontend/src/app/App.tsx:62`](frontend/src/app/App.tsx#L62), so tabs cannot remain hidden.
- `showTabSelector` is unused state at [`frontend/src/app/App.tsx:58`](frontend/src/app/App.tsx#L58).
- A localStorage failure can prevent state update and leave the app at a permanent blank return path around [`frontend/src/app/App.tsx:89`](frontend/src/app/App.tsx#L89).
- Local connection says the granted root can be selected, but `canConnect` requires a non-empty subfolder at [`frontend/src/app/pages/ConnectPage.tsx:116`](frontend/src/app/pages/ConnectPage.tsx#L116).
- [`frontend/src/app/pages/ExportPage.tsx`](frontend/src/app/pages/ExportPage.tsx) is unrouted/dead and uses raw sources instead of canonical source keys.

**Fix**

- Add rendered behavior tests for customization, storage failure, and root selection.
- Keep in-memory state updates independent of storage success and surface persistence warnings.
- Remove or reconnect dead features; avoid maintaining two divergent export implementations.

### M-13 — Accessibility coverage and keyboard behavior are incomplete

Observed issues include:

- modals without complete `role="dialog"`/`aria-modal`, focus trap/restoration, Escape handling, or inert background behavior;
- canvas global keyboard handlers remaining active behind modals;
- mouse-only resize separators at [`frontend/src/components/Browse/ResizeDivider.tsx:71`](frontend/src/components/Browse/ResizeDivider.tsx#L71);
- a dropzone with `role="button"` but no Enter/Space handler at [`frontend/src/components/Ingest/IngestDropzone.tsx:375`](frontend/src/components/Ingest/IngestDropzone.tsx#L375);
- important controls hidden with hover-only opacity at [`frontend/src/components/Browse/LocalSampleBrowser.tsx:197`](frontend/src/components/Browse/LocalSampleBrowser.tsx#L197) and [`frontend/src/components/annotate/VersionHistoryModal.tsx:117`](frontend/src/components/annotate/VersionHistoryModal.tsx#L117);
- icon-only controls without reliable accessible names.

**Fix**

Build/reuse an accessible modal primitive, stop background shortcuts while overlays are open, implement keyboard separator/dropzone behavior, use `focus-visible` styles, and add Testing Library plus axe tests. Perform a manual keyboard/screen-reader pass on the core workflow.

### M-14 — Backend packaging, module size, and configuration boundaries need redesign

**Evidence**

- [`backend/annotation_server.py`](backend/annotation_server.py) is 1,283 lines and owns all 33 routes plus policy, filesystem decisions, job creation, and side effects.
- [`frontend/src/components/annotate/AnnotationCanvas/index.tsx`](frontend/src/components/annotate/AnnotationCanvas/index.tsx) is 2,802 lines. Other 500–600 line UI modules combine API orchestration and rendering.
- [`backend/pyproject.toml:34`](backend/pyproject.toml#L34) sets `py-modules = []`; Docker installs an empty distribution then relies on working-directory imports.
- Configuration is read through scattered import-time globals. Used settings such as `TILED_BROWSE_PATH`, `BROWSE_FIELD_MAPPING_TTL_SECONDS`, and `EXPORT_WORKERS` are absent from `.env.example`.
- The frontend contains 27 scattered raw `fetch()` calls with inconsistent status, cancellation, and error behavior.

**Impact**

Cross-cutting security rules cannot be enforced in one place, unit isolation is difficult, settings can be invalid until runtime, and multi-worker/test behavior depends on import order and current working directory.

**Fix**

- Create a real installable `src/segmentation_annotation_studio/` package with an app factory.
- Split thin routers from application services, repositories, job orchestration, and Tiled/local-source gateways.
- Centralize validated settings with `pydantic-settings` and keep `.env.example` generated/complete.
- Split the canvas into identity-safe image loading, rendering layers, tool state machines, shape editing, and SAM/wand controllers.
- Generate a typed frontend client from response-modeled OpenAPI and centralize timeouts, cancellation, errors, and authentication.

### M-15 — CI and tests miss the boundaries where the serious bugs live

**Evidence**

- CI only runs the checks shown at [`.github/workflows/ci.yml:19`](.github/workflows/ci.yml#L19) and [`.github/workflows/ci.yml:33`](.github/workflows/ci.yml#L33).
- The application has 33 FastAPI routes, but only two direct HTTP-boundary calls were found in `test_health.py`.
- There are 37 TSX files and zero `.test.tsx` files; the 89 frontend tests focus overwhelmingly on pure algorithms.
- No E2E suite, coverage threshold, frontend lint, Python type-check, strict docs build, Docker build, shell/PowerShell check, or automated security scan is enforced.
- CI uses Python 3.11 while Docker/local uses 3.12; Node versions are also inconsistent.
- Unpinned lint tools already drift: one valid older flake8/isort combination fails imports that a fresh current environment accepts. Black 26.5.1 reports 30 files requiring formatting.
- [`frontend/tsconfig.app.json:16`](frontend/tsconfig.app.json#L16) disables unused checks, and no ESLint setup exists.

**Fix**

- Add API contract/security tests for auth, unknown Tiled servers, secret-free OpenAPI, filesystem roots, traversal, limits, cleanup, concurrency, cache headers, and error redaction.
- Add Testing Library/MSW/fake-timer tests for image transitions, autosave errors/order, guide sync, export, undo, routing, and accessibility.
- Add Playwright smoke flows: connect, open, annotate, switch slice, save, restore, export, and failure recovery.
- Add coverage thresholds, frontend build/lint, Python type checking, formatting, docs strict build, Docker build, and dependency/security jobs.
- Test Python 3.11/3.12 and one documented Node version; pin tools or consolidate on a well-configured Ruff/Black toolchain.
- Pin GitHub Actions to reviewed commit SHAs and set minimal workflow permissions.

### M-16 — Container hardening, portability, and documentation drift reduce deployment reliability

**Evidence**

- Docker uses mutable base tags, runs as root, and defines no healthcheck at [`Dockerfile`](Dockerfile).
- Compose lacks health, resource, restart, read-only filesystem, `no-new-privileges`, and capability controls.
- Build revision is likely always `dev` because Vite looks for Git metadata but the frontend build stage receives only the frontend directory.
- License/copyright files are not copied into the final image.
- [`tiled/config.yml:17`](tiled/config.yml#L17) contains a developer-specific `/Users/david/...` path.
- [`.tiled/README.md:3`](.tiled/README.md#L3) claims a bundled catalog/data set, but this snapshot contains only that README in `.tiled`.
- Installation docs still contain Windows/WSL drift despite a native launcher.
- The production Docs URL defaults to a local MkDocs address that the container does not serve.

**Fix**

- Pin base patch releases/digests with automated updates; run as a non-root UID with only `/data` writable.
- Add health/readiness checks, OCI metadata, licensing notices, dropped capabilities, `no-new-privileges`, and sensible resource/restart policies.
- Pass the source revision explicitly at build time.
- Remove personal paths and use validated environment/project-relative roots.
- Correct bundled-data, Windows, docs-hosting, and Node-version documentation.

### M-17 — Local secret/data permissions and repository governance need hardening

**Evidence**

- [`start_all.sh:381`](start_all.sh#L381) writes `.env` without explicitly enforcing `0600`; draft/version/guide directories rely on the user's umask.
- A retired credential-shaped full literal remains in both launchers around [`start_all.sh:401`](start_all.sh#L401) and the corresponding Windows code. It appears to be legacy detection/rotation logic, but should not remain in source or secret-scanner output.
- Missing project controls include a `SECURITY.md`, dependency-update configuration, CODEOWNERS, contributing/change policy, `.editorconfig`, `.gitattributes`, and pre-commit configuration.

**Impact**

On shared systems, local annotations or secrets may receive broader permissions than intended. Historical credential material can remain valid in deployments that bypass rotation and causes secret-scanning ambiguity. Missing governance slows responsible disclosure and consistent maintenance.

**Fix**

- Create secret/data directories as `0700` and files as `0600`, with platform-appropriate Windows ACLs.
- Revoke the legacy credential at its source and replace source comparison with a non-reversible fingerprint/migration marker. Purge history only through a coordinated repository-history rewrite.
- Add `SECURITY.md` with supported versions/reporting, automated dependency updates, ownership, contribution/release policy, and canonical dev/CI commands.

---

## Project structure assessment

### Verdict

The project is **well structured for a small local scientific prototype**, but **not yet well structured for a secure production service**. The top-level separation is clear and the domain vocabulary is coherent. The key problem is that trust-boundary policy and stateful orchestration are concentrated in a few oversized modules and process-global objects.

### Scorecard

| Area | Assessment | Notes |
|---|---|---|
| Top-level repository layout | Good | Backend, frontend, docs, Tiled, Docker, and platform launchers are easy to find |
| Domain decomposition | Good/partial | Backend image/export/draft/guide/Tiled logic is split; route/policy layer is monolithic |
| Frontend organization | Good/partial | Pages/hooks/stores/libs are clear; canvas and ingest/browser components are oversized |
| API/security boundaries | Poor | No auth, caller-selected Tiled/filesystem targets, duplicated policy, raw dictionaries |
| Backend packaging | Poor | Empty distribution, flat generic modules, working-directory imports |
| State/persistence | Poor/partial | Simple local files suit a prototype; concurrency, versioning, migrations, and jobs do not |
| Type/contracts | Partial | Strict TypeScript and shape types exist; backend boundaries bypass them and contracts are duplicated |
| Tests | Partial | Strong geometry/export unit tests; little route/component/workflow/security coverage |
| Dependency reproducibility | Poor | npm lock exists; Python/model/bootstrap paths are mutable or divergent |
| Deployment | Poor | Exposure, secret build context, root container, no production auth/hardening |
| Documentation | Good/partial | Thorough architecture/user docs, with several important security/runtime inconsistencies |

### Recommended target layout

The exact names can vary, but the dependency direction should be explicit:

```text
backend/
  pyproject.toml
  src/segmentation_annotation_studio/
    app.py                     # app factory, lifespan, middleware
    settings.py                # one validated settings model
    api/
      dependencies.py         # identity, authorization, source lookup
      routers/
        config.py
        browse.py
        images.py
        annotations.py
        guides.py
        exports.py
        ingest.py
    domain/                    # annotation/export models and invariants
    services/                  # application workflows, no HTTP concerns
    repositories/             # transactional drafts/versions/guides/jobs
    gateways/
      tiled.py                 # allowlisted server IDs and credentials
      local_sources.py         # opaque roots/sources and containment
    workers/                   # bounded/durable job definitions
  tests/
    unit/
    integration/
    security/
    fixtures/

frontend/src/
  api/                         # generated contracts + one request client
  features/
    connect/
    browse/
    annotate/
      canvas/
        imageIdentity.ts
        layers/
        tools/
        segmentation/
      persistence/
    export/
    reference/
  stores/                      # source-scoped editor/document state
  components/                  # shared accessible primitives
  test/                        # MSW fixtures and helpers
```

Important architectural rules:

1. Routes accept IDs and validated models, not arbitrary network URIs or server paths.
2. One dependency/gateway authorizes a source before any read, render, export, or write.
3. Domain services cannot access global credentials or process environment implicitly.
4. Writes use revisions, transactions/staging, and explicit operation authorization.
5. Frontend editor state is source-scoped and one atomic snapshot defines undo/save/export.
6. OpenAPI response models generate the TypeScript boundary contract.
7. Background jobs are bounded, observable, cancelable, and durable where production requires it.

### Large-module decomposition priorities

| Module | Approx. lines | First extraction |
|---|---:|---|
| `AnnotationCanvas/index.tsx` | 2,802 | Identity-safe image loader, render layers, pointer/tool state machines, SAM/wand controller |
| `annotation_server.py` | 1,283 | App factory, feature routers, shared auth/source dependencies, job services |
| `coco_export.py` | 619 | Validated export plan, safe naming/path layer, format writers |
| `IngestDropzone.tsx` | 590 | Upload state machine, validation, progress UI, conflict workflow |
| `browse_helpers.py` | 534 | Query/filter validation, Tiled traversal, response mapping |
| `ColumnBrowser.tsx` | 506 | Query controller, column views, virtualized list primitives |
| `ConnectPage.tsx` | 502 | Local/Tiled connection forms and root selection |
| `annotationStore.ts` | 478 | Atomic source document state versus ephemeral tool/session state |

---

## Test and quality assessment

### What is good

- Backend rasterization, exports, schema helpers, drafts, paths, masks, and numerical utilities have meaningful tests.
- Frontend geometry, raster processing, morphology, and image utilities have strong pure-function coverage.
- TypeScript strict mode is enabled.
- `np.load(..., allow_pickle=False)` and `yaml.safe_load()` avoid common unsafe deserialization classes.
- The frontend uses the backend API rather than calling Tiled directly.

### What must be added

| Boundary | Required tests |
|---|---|
| Authentication/authorization | Every read/write route, role/scope matrix, CSRF/Origin/Host behavior |
| Tiled gateway | Unknown server rejection, no default key leakage, redirects/timeouts, cache partition |
| Filesystem | Arbitrary root/absolute path rejection, symlink containment, opaque source authorization |
| Export | Absolute/`..`/sibling-prefix names, ZIP safety, conflict atomicity, collisions, multi-source taxonomy |
| Ingest/mask writes | Limits, corrupt replacement preserves old data, rollback, concurrency, confirmation token |
| Annotation schemas | NaN/Infinity, normalization, huge geometry, duplicate classes, invalid refs/splits/ratios |
| Persistence | Concurrent saves, ETag conflicts, immutable versions, migrations, guide exclusion |
| Frontend image identity | Slow/out-of-order/error loads, source/slice change, SAM/wand cache identity |
| Autosave | 404 vs 500/network, ordering, unmount/unload, retries, conflict recovery |
| Editor history | Classes/shapes/splits/negatives, restore transactions, first-edit dirty state |
| Accessibility | Dialog focus, keyboard-only core flow, axe scan, hidden actions |
| E2E | Connect → browse → annotate → switch → save → restore → export; failure/recovery flow |

### Suggested CI gates

1. Locked dependency installation.
2. Python format/lint/type-check on 3.11 and 3.12.
3. Backend unit, integration, and security tests with coverage.
4. Frontend lint/type-check/unit/component tests with coverage.
5. Production frontend build and bundle budget.
6. Playwright smoke suite.
7. MkDocs strict build, shell/PowerShell checks, and Docker build/health smoke test.
8. Secret scan, dependency audit, SAST, SBOM, and container scan.
9. Minimal workflow permissions and commit-pinned third-party actions.

---

## Remediation roadmap

### Phase 0 — Containment (same day)

- Fix both Vite launcher hosts and Compose port binding.
- Add `.env`/model build-context rules; rotate any possibly baked key.
- Disable external/network deployment or privileged Tiled writes until access control exists.
- Reject unconfigured Tiled URIs and remove client/query key plumbing.
- Warn users against valuable-data replace operations and against trusting “no data leaves” until feedback behavior changes.

### Phase 1 — Security and data-integrity boundary (first week)

- Implement identity/authentication, scoped authorization, Host/Origin/CSRF controls, and write auditing.
- Add allowlisted `TiledGateway` and opaque `LocalSourceRegistry`.
- Fix safe export paths/archive names and stage all outputs.
- Add request/decoded-data/geometry quotas and a bounded job queue.
- Make Tiled replace/mask publication non-destructive and revision-aware.
- Fix image identity, draft/guide load state, multi-source export identifiers/taxonomy, and atomic editor state.
- Add regression tests for every Critical and High finding.

### Phase 2 — Transactional persistence and reproducibility (weeks 2–4)

- Move drafts, guides, versions, and job metadata into transactional/versioned storage.
- Add ETags/revisions, immutable version constraints, schema migrations, and recovery copies.
- Package the backend properly and split routers/services/gateways.
- Lock Python/bootstrap/model artifacts; update npm dependencies and automate updates.
- Introduce generated API contracts and one frontend client.
- Add component/API/E2E/coverage gates and standardize tool/runtime versions.

### Phase 3 — Production hardening and maintainability

- Run the container non-root with health checks, minimal privileges, resource limits, immutable artifacts, and complete provenance/licensing.
- Add CSP/security headers, feedback consent/privacy controls, and accessible shared UI primitives.
- Decompose large modules, virtualize large lists, and add performance/memory budgets.
- Correct docs/config drift and add security/contribution/release governance.

---

## Positive controls and strengths

The audit did **not** identify a request-reachable SQL injection, shell-command injection, `eval`/`exec` path, pickle deserialization, unsafe YAML object construction, or obvious React HTML-injection sink.

Specific strengths worth preserving:

- `/api/config/servers` deliberately omits API keys and has a regression test.
- Local FastAPI and Tiled launcher commands themselves use `127.0.0.1`; the exposure comes from the Vite CLI override.
- Default CORS origins are explicit and credentials are disabled.
- Path containment under a genuinely server-owned root uses resolved paths rather than a naive string prefix.
- Upload temporary filenames are server-generated.
- `np.load(..., allow_pickle=False)` and `yaml.safe_load()` are used.
- Request-path database code does not interpolate raw SQL; maintenance scripts parameterize statements.
- Tiled ingest error URLs receive some sanitization.
- No hardcoded active Tiled key was found in frontend source, and explicit frontend API calls target the backend rather than Tiled.
- React escaping is preserved; no `dangerouslySetInnerHTML` sink was found.
- Lazy pages, a SAM worker, strict TypeScript, normalized rectangle/ellipse behavior, and Phosphor icon use are good implementation foundations.
- The domain-specific geometry/raster/export test investment is valuable.

---

## Definition of done for a network-capable release

A release should not be described as production-ready until all of the following are demonstrably true:

- [ ] No service is unintentionally published; network mode is explicit.
- [ ] Every API operation has an authenticated identity and tested authorization policy.
- [ ] Tiled destinations and local roots are server-owned allowlisted records, never arbitrary request values.
- [ ] No credential appears in frontend state, query strings, OpenAPI parameters, logs, images, or build layers.
- [ ] Destructive operations require server-verified intent and preserve a recoverable prior revision.
- [ ] Export/ingest/mask writes are staged, validated, conflict-safe, and traversal-safe.
- [ ] Request, decoded-image, geometry, concurrency, and job-retention limits are enforced.
- [ ] Drafts/guides/versions use revisions, transactions, schema versions, and tested migrations.
- [ ] The rendered image identity is cryptographically/logically tied to the slice that receives edits.
- [ ] Multi-source export has reversible identifiers and validated per-source/shared taxonomy semantics.
- [ ] Python, Node, frontend, model, installer, and container artifacts are reproducible and pinned.
- [ ] Critical workflows have API, component, E2E, accessibility, concurrency, and recovery tests.
- [ ] Privacy documentation accurately covers feedback, model downloads, logging, and external services.
- [ ] Container runtime is non-root, minimally privileged, health-checked, and scanned.

## Audit limitations

- The supplied workspace has no usable `.git` directory, so tracked-versus-ignored state, commit discipline, history, and the documented historical secret could not be independently verified. No audit commit could be created.
- No live production infrastructure, reverse proxy, identity provider, registry, or cloud egress policy was in scope.
- No Tiled mutation or destructive exploit was executed; write-path conclusions come from source/control-flow analysis and existing tests.
- npm advisories are time-sensitive and were checked on the audit date. Their contextual reachability was reviewed, but a full transitive SCA/container CVE platform was not available.
- This is a detailed engineering/security review, not a formal compliance certification or exhaustive penetration test.

## Priority index

| ID | Short title | Owner area | First verification |
|---|---|---|---|
| C-01 | Vite LAN exposure | Launchers/frontend | Listener smoke test |
| C-02 | Anonymous published privileged API | Platform/backend | Auth matrix and remote denial test |
| C-03 | Tiled SSRF/key exfiltration | Backend/Tiled | Capture-server negative test |
| C-04 | `.env` copied into Docker | Platform | Build-context/image secret scan |
| H-01 | Caller-selected filesystem root | Backend | Absolute/root/symlink tests |
| H-02 | Export traversal/ZIP safety | Backend/export | Malicious-name test matrix |
| H-03 | Delete-before-validate Tiled writes | Backend/Tiled | Failure-injection preservation test |
| H-04 | Resource exhaustion | Backend/platform | Quota and stress tests |
| H-05 | Query-string key plumbing | Full stack | OpenAPI/request/log assertions |
| H-06 | Concurrent persistence overwrite | Backend/storage | Concurrent save/version tests |
| H-07 | Stale image writes to new slice | Frontend | Reordered/failing image tests |
| H-08 | Load failure followed by overwrite | Full stack | 500/network/404 autosave tests |
| H-09 | Broken/mislabeled multi-export | Frontend/export | Multi-source round-trip fixture |
| H-10 | Annotation validation bypass | Backend/contracts | Fuzzed boundary validation tests |
| H-11 | Incoherent undo/save state | Frontend/state | Compound history/dirty tests |

