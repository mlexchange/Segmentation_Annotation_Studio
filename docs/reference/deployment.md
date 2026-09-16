# Production deployment

This page is for standing up a shared, long-lived deployment (e.g. behind
`hub.als.lbl.gov`), as opposed to [Installation](../getting-started/installation.md)'s
"run it yourself" Docker options. It covers the published container images, hosting
under a URL prefix, and what's still an open question before a real production rollout.

---

## Published images

Every merge to `main` builds and publishes images to GitHub Container Registry
(`ghcr.io`) via its own dedicated workflow, `.github/workflows/publish-image.yml` —
this never fires on a pull request, only on `main` (PRs get a build-only check
instead, in `ci.yml`'s `docker-build` job — same Dockerfile targets, no registry
push). The structure mirrors
[als-computing/view_tomography_recon_app](https://github.com/als-computing/view_tomography_recon_app)'s
own `publish-image.yml` — same tag names, same one-image-many-tags shape, so anyone
already familiar with that repo's deployment finds this one working the same way.

**One image name, three tags** — the Dockerfile target differs per tag (invisible
from the registry, but real underneath):

| Tag | Dockerfile target | Tiled | iPred / Train | Base path baked in | Compose file |
| --- | --- | --- | --- | --- | --- |
| `:local` | `app-full` | Bundled | Bundled | `/seg_studio/` | `docker-compose.local.yml` |
| `:als-prod` | `app-ml` | External (ALS's production Tiled) | Bundled | `/bl832/seg_studio/` | `docker-compose.als-prod.yml` |
| `:als-staging` | `app-ml` | External (ALS's staging Tiled) | Bundled | `/bl832/seg_studio/` | `docker-compose.als-staging.yml` |

Pull with `ghcr.io/<owner>/<repo>:local` / `:als-prod` / `:als-staging`.

`:als-prod` and `:als-staging` build identically today — this app reads `TILED_URI`
at container-*run* time (an ordinary env var), not at build time, so the
prod-vs-staging Tiled distinction lives entirely in which compose file's default you
use, not in the image itself. They're still published as two separate, explicitly
named steps in `publish-image.yml` (not one build tagged twice) so they can diverge
later — e.g. if a build-time-only setting ever needs to differ between the two —
without restructuring the workflow.

`:local` is fully self-contained (Tiled + backend + iPred, nothing external
required) for anyone standing this up for themselves who still wants to exercise the
same subpath-hosting path `:als-prod`/`:als-staging` use, rather than leaving that
path tested only at ALS — see [Installation](../getting-started/installation.md) for
running it locally. `:als-prod`/`:als-staging` bundle iPred/Train (so they work
without a second, separately-maintained service) but keep Tiled external, since ALS
already runs its own Tiled and a second, empty in-container one would be wrong.

!!! note "No `:latest`/`-sha` tags — a deliberate simplification"
    Earlier in this repo's history, every image also published generic `:latest` +
    immutable `:sha-<commit>` tags alongside the purpose-named ones. That's been
    dropped in favor of exactly the three tags above, matching the reference repo's
    own simpler scheme — one clearly-named tag per real deployment target, nothing
    else. The trade-off: there's no immutable per-commit tag to roll back to
    anymore; rolling back today means re-running `publish-image.yml` against an
    older commit (`git push -f` a maintenance branch to `main`, or a manual
    workflow dispatch pinned to a specific ref) rather than just pointing at a
    `-sha` tag that's already sitting in the registry.

---

## Hosting under a URL prefix

A shared deployment is typically reached at a path prefix off a shared hub domain
(e.g. `hub.als.lbl.gov/bl832/seg_studio/`) rather than its own subdomain, so it stays
manageable from an ops/ticketing perspective. This requires two things working together:

1. **A reverse proxy in front of the container that strips the prefix** before
   forwarding — the app container itself has no idea it's hosted under a subpath; it
   only ever serves and expects plain root-relative paths (`/api/...`, `/assets/...`).
   A request to `hub.als.lbl.gov/bl832/seg_studio/api/...` must reach the container as
   plain `/api/...`. This is standard nginx/Traefik path-based routing — the backend
   itself needs **no code changes** for this.
2. **The frontend build baked for that prefix**, via the `VITE_BASE_PATH` Docker
   build-arg (e.g. `--build-arg VITE_BASE_PATH=/bl832/seg_studio/`). Vite bakes this
   into the compiled JS at build time (`base` in `vite.config.ts`, consumed by
   `main.tsx`'s router `basename` and `config.ts`'s `API_BASE`) — **root-hosted and
   subpath-hosted are two different build artifacts from the same source**, not one
   image that adapts at runtime. The published `:als-prod`/`:als-staging`/`:local`
   tags already have this baked in — building your own image (e.g. `docker-
   compose.yml`/`docker-compose.ml.yml`/`docker-compose.full.yml`'s generic,
   unpublished targets) with `VITE_BASE_PATH` left unset stays root-hosted.

If you're standing up your own subpath deployment (a different institution, a
different path), build your own image with the matching `VITE_BASE_PATH` rather than
reusing the `:als-prod`/`:als-staging`/`:local` tags, which are baked for this repo's specific paths.

!!! warning "`VITE_BASE_PATH` is a path, never a hostname"
    Set it to the path segment only (`/bl832/seg_studio/`) — **never** a full URL
    with a scheme or host (`https://hub.als.lbl.gov/bl832/seg_studio/` would be
    wrong). The image has no business knowing what domain fronts it; only the
    reverse proxy does, and it can change (or differ between environments)
    without ever touching this build-arg. This is what makes the *same* `:als-prod`/`:als-staging`
    image usable unmodified across staging and production — e.g.
    `hub-staging.als.lbl.gov/bl832/seg_studio/` and
    `hub.als.lbl.gov/bl832/seg_studio/` sharing the identical `/bl832/seg_studio/`
    path, routed to whichever environment's container by the proxy's own
    hostname-based rule, not by anything baked into the image. Baking a real
    FQDN in would force a separate image build per environment for no reason,
    and would silently break if that hostname ever changed.

!!! tip "Testing this locally before it matters"
    `docker-compose.local.yml` (see [Installation](../getting-started/installation.md))
    includes a small nginx service that reproduces exactly this proxy-stripping
    behavior, so `/seg_studio/` hosting can be verified on a laptop before it's ever
    tried against a real hub deployment. Serving the built `dist/` directly through a
    plain static file server does **not** reproduce this — the stripping proxy is
    load-bearing, not optional.

!!! warning "A real reverse proxy needs a body-size limit raised too"
    Discovered via the local test proxy: nginx's own default `client_max_body_size`
    (1MB) rejects the Ingest tab's multi-file uploads (a folder of TIFFs easily
    reaches hundreds of MB) with a raw `413` *before the request ever reaches this
    app* — this app's own upload handling is never even consulted. `docker/nginx-
    local.conf` sets `client_max_body_size 0;` (unlimited; the backend's own
    job-based, streamed-to-disk ingest route is the real, size-aware boundary) —
    **`hub.als.lbl.gov`'s real reverse proxy needs the equivalent setting**, or
    every real-world upload through this app will hit the same wall in production.

---

## Environment variables specific to a shared deployment

Beyond what's already covered in [Installation](../getting-started/installation.md#environment-variables):

| Variable | Purpose |
| --- | --- |
| `TILED_BROWSE_PATH` | The real path into an existing institutional Tiled catalog Browse should treat as its root (e.g. `beamlines/bl832/processed`). Confirm the exact value with whoever operates that Tiled server — don't assume it matches another deployment's beamline. |
| `TILED_URI` / `TILED_API_KEY` | Point at the shared production Tiled server. See the open authentication question below — a single shared key is a stopgap, not the final design. |
| `LOCAL_SOURCE_DIR` | `docker-compose.full.yml`/`docker-compose.local.yml` (bundled-Tiled shapes) only — bind-mounts a real host directory to `/data/processed` so the bundled Tiled server can read it directly. Everything under it (Zarr stores and plain image-slice folders) auto-registers into Tiled on every container start (see [Installation](../getting-started/installation.md#option-2-docker)) — no manual ingest step needed for pre-existing data. Not applicable to `:als-prod`/`:als-staging` (`app-ml`), which point at an already-existing external Tiled instead of a bundled one. |

Persistent storage (`LOCAL_DATA_ROOT`, defaulting to the `/data` volume already
declared in the image) and CORS (`BROWSE_ALLOWED_ORIGINS`) need no special
production-specific handling beyond what's already documented for local Docker use —
mount a real volume, and leave CORS empty since the SPA and API are served same-origin.

### Setting these for `docker compose`

Copy `.env.example` (repo root) to `.env` and fill it in — `docker compose` loads a
`.env` file in the same directory automatically, for every compose file:

```bash
cp .env.example .env
# edit .env: set TILED_API_KEY at minimum — TILED_URI/TILED_BROWSE_PATH/
# VITE_BASE_PATH already default correctly per environment (see below)
docker compose -f docker-compose.als-prod.yml up --build
```

For ALS specifically, use `docker-compose.als-prod.yml`/`docker-compose.als-
staging.yml` — these already default `TILED_URI` to the real ALS production/staging
Tiled hostnames and `VITE_BASE_PATH` to `/bl832/seg_studio/`, so only
`TILED_API_KEY` needs setting per deployment. `docker-compose.ml.yml` is the
generic, no-institution-specific-defaults version of the same shape (any other
Tiled server, any base path) — set `TILED_URI`/`TILED_BROWSE_PATH`/`VITE_BASE_PATH`
yourself there.

`.env` is git-ignored — never commit a real `TILED_API_KEY` into it. This is a
different file from `backend/.env.example`/`frontend/.env.example`, which configure
`start_all.sh`/plain `npm run dev`/`npm run build` outside a container — the root
`.env` is specifically what `docker compose` itself substitutes into the compose
files' `${VAR}` references. If you're running the already-published `ghcr.io` image
directly (`docker run`/your own orchestration) rather than through one of these
compose files, pass the equivalent `-e`/env vars there instead — `.env` only affects
`docker compose` invocations in this directory.

---

## Open questions before a real production rollout

These are real, unresolved gaps — not yet implemented, flagged here rather than
glossed over:

- **Per-user Tiled authentication.** Tiled's authorization is per-proposal
  (ESAF) read/write tags, not a blanket grant — a single shared `TILED_API_KEY`
  either over- or under-privileges every user. The real design needs per-user
  ORCID/OIDC login (for both the browser's direct-to-Tiled 3D viewer calls and the
  backend's own proxied calls), which hasn't been built yet. Until then, a shared
  `TILED_API_KEY` works but does not correctly represent per-proposal permissions —
  fine for evaluation, not for a real multi-user production rollout with real data.
- **Hub-level SSO.** Whether `hub.als.lbl.gov`'s own reverse proxy fully gates access
  before a request reaches this app, or whether the app is expected to participate in
  auth itself, is a separate open question from Tiled's own data-access control above
  — confirm with hub admins before relying on either assumption.
- **GPU availability for `:als-prod`/`:als-staging`.** Bundling iPred/ML there only delivers a
  practically usable Train tab if the host running that container provides GPU
  passthrough — CPU-only training/inference works but is dramatically slower.
