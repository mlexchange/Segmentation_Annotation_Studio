# Production deployment

This page is for standing up a shared, long-lived deployment (e.g. behind
`hub.als.lbl.gov`), as opposed to [Installation](../getting-started/installation.md)'s
"run it yourself" Docker options. It covers the published container images, hosting
under a URL prefix, and what's still an open question before a real production rollout.

---

## Published images

Every merge to `main` builds and publishes images to GitHub Container Registry
(`ghcr.io`) via `.github/workflows/ci.yml`'s `docker-publish` job — this never fires
on a pull request, only on `main`.

| Image | Tags | Dockerfile target | Tiled | iPred / Train | Base path baked in |
| --- | --- | --- | --- | --- | --- |
| `ghcr.io/<repo>/app` | `:latest`, `:sha-<commit>` | `app` | External | Not included | None (root-hosted) |
| `ghcr.io/<repo>/app-ml` | `:latest`, `:sha-<commit>` | `app-ml` | External | Bundled | None (root-hosted) |
| `ghcr.io/<repo>/app-ml` | `:als`, `:als-<commit>` | `app-ml` | External (ALS's own production Tiled) | Bundled | `/bl832/seg_studio/` |
| `ghcr.io/<repo>/app-full` | `:latest`, `:sha-<commit>` | `app-full` | Bundled | Bundled | None (root-hosted) |
| `ghcr.io/<repo>/app-full` | `:local`, `:local-<commit>` | `app-full` | Bundled | Bundled | `/seg_studio/` |

Every tag has an immutable `-<commit-sha>` sibling, so pinning a deployment to a
specific known-good build (rather than a floating tag) is always available —
that's the rollback story: point at the previous `-sha` tag.

`:als` is the shape ALS's own hub deployment uses: iPred/Train bundled (so they work
without a second, separately-maintained service), Tiled kept external since ALS
already runs its own production Tiled and a second, empty in-container one would be
wrong. `:local` is fully self-contained (Tiled + backend + iPred, nothing external
required) for anyone standing this up for themselves who still wants to exercise the
same subpath-hosting path `:als` uses, rather than leaving that path tested only at
ALS — see [Installation](../getting-started/installation.md) for running it locally.

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
   image that adapts at runtime. The published `:als`/`:local` tags already have this
   baked in; the generic `:latest`/`:sha` tags of every image are root-hosted and
   unaffected.

If you're standing up your own subpath deployment (a different institution, a
different path), build your own image with the matching `VITE_BASE_PATH` rather than
reusing the `:als`/`:local` tags, which are baked for this repo's specific paths.

!!! warning "`VITE_BASE_PATH` is a path, never a hostname"
    Set it to the path segment only (`/bl832/seg_studio/`) — **never** a full URL
    with a scheme or host (`https://hub.als.lbl.gov/bl832/seg_studio/` would be
    wrong). The image has no business knowing what domain fronts it; only the
    reverse proxy does, and it can change (or differ between environments)
    without ever touching this build-arg. This is what makes the *same* `:als`
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

---

## Environment variables specific to a shared deployment

Beyond what's already covered in [Installation](../getting-started/installation.md#environment-variables):

| Variable | Purpose |
| --- | --- |
| `TILED_BROWSE_PATH` | The real path into an existing institutional Tiled catalog Browse should treat as its root (e.g. `beamlines/bl832/processed`). Confirm the exact value with whoever operates that Tiled server — don't assume it matches another deployment's beamline. |
| `TILED_URI` / `TILED_API_KEY` | Point at the shared production Tiled server. See the open authentication question below — a single shared key is a stopgap, not the final design. |

Persistent storage (`LOCAL_DATA_ROOT`, defaulting to the `/data` volume already
declared in the image) and CORS (`BROWSE_ALLOWED_ORIGINS`) need no special
production-specific handling beyond what's already documented for local Docker use —
mount a real volume, and leave CORS empty since the SPA and API are served same-origin.

### Setting these for `docker compose`

Copy `.env.example` (repo root) to `.env` and fill it in — `docker compose` loads a
`.env` file in the same directory automatically, for every compose file:

```bash
cp .env.example .env
# edit .env: set TILED_URI, TILED_API_KEY, TILED_BROWSE_PATH, VITE_BASE_PATH
docker compose -f docker-compose.ml.yml up --build
```

`.env` is git-ignored — never commit a real `TILED_API_KEY` into it. This is a
different file from `backend/.env.example`/`frontend/.env.example`, which configure
`start_all.sh`/plain `npm run dev`/`npm run build` outside a container — the root
`.env` is specifically what `docker compose` itself substitutes into the compose
files' `${VAR}` references (`TILED_URI`, `TILED_API_KEY`, and `TILED_BROWSE_PATH` in
`docker-compose.yml`/`docker-compose.ml.yml`; `VITE_BASE_PATH` as a build-arg in
`docker-compose.ml.yml`). If you're running the already-published `ghcr.io` image
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
- **GPU availability for `:als`.** Bundling iPred/ML into `:als` only delivers a
  practically usable Train tab if the host running that container provides GPU
  passthrough — CPU-only training/inference works but is dramatically slower.
