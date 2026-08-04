# Windows launcher

Native Windows equivalent of the repo's `start_all.sh`. Starts Tiled, the backend API,
the frontend, and the docs site together — bootstrapping everything on first run — on
stock Windows 10/11 with **no WSL and no Git Bash**.

## Prerequisites

- **Node.js 20+** (`npm`) — install from <https://nodejs.org> or `winget install OpenJS.NodeJS`.
- **PowerShell 5.1+** — ships with Windows 10/11 (PowerShell 7 also works).

Everything else (`uv`, a Python 3.12 `.venv`, backend dependencies, the SlimSAM model,
the Tiled API key) is installed/generated automatically on first run.

## Run it

Easiest — **double-click** `windows\start_all.cmd`.

Or from a terminal, in the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\start_all.ps1
```

When it's ready it prints the URLs (Frontend, Backend, Tiled, Docs). Open the **Frontend**
URL. Press **Ctrl+C** in the window to stop all services.

### Production build (single origin)

Build the optimized SPA and have the backend serve it (no Vite dev server); the whole app
is then at the **Backend** URL:

```powershell
$env:PROD = "1"; .\windows\start_all.ps1
```

### Overriding ports

Ports auto-fall back to the next free one if a default is busy. To force specific ports:

```powershell
$env:BACKEND_PORT = "9002"; $env:FRONTEND_PORT = "5273"; .\windows\start_all.ps1
```

(`TILED_PORT`, `BACKEND_PORT`, `FRONTEND_PORT`, `DOCS_PORT` are all honored.)

## Notes & troubleshooting

- **Execution policy**: `start_all.cmd` and the `-ExecutionPolicy Bypass` invocation both
  bypass the policy for that one process only — no system-wide change.
- **`uv` just installed but "not found"**: open a **new** PowerShell window (so `%USERPROFILE%\.local\bin`
  is on `PATH`) and re-run.
- **`pycocotools` build error**: recent versions ship Windows wheels; if the install fails,
  install the **"Desktop development with C++"** workload from the Visual Studio Build Tools
  and re-run, or the launcher will retry the dependency install.
- **Behavior parity**: this is a 1:1 port of `../start_all.sh` — same ports, same auto
  port-fallback, same Tiled key handling (anonymous access is read-only; the generated key
  authenticates writes and is never sent to the browser), same `PROD=1` prod-serving mode.
  All services bind to `127.0.0.1` only.
