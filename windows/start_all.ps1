#Requires -Version 5.1
<#
  start_all.ps1 - Windows launcher for Segmentation Annotation Studio.

  Native PowerShell port of ../start_all.sh. Starts Tiled, the backend API, the
  frontend, and the docs site together, bootstrapping everything on first run
  (uv -> .venv -> deps -> Tiled API key -> SAM model). Runs on stock Windows
  (PowerShell 5.1+); no WSL or Git Bash required.

  Usage (from anywhere):
      powershell -ExecutionPolicy Bypass -File .\windows\start_all.ps1
  or just double-click windows\start_all.cmd.

  Stop everything: Ctrl+C.

  Options (environment variables):
      $env:PROD = "1"            build the optimized SPA; the backend serves it
      $env:TILED_PORT / $env:BACKEND_PORT / $env:FRONTEND_PORT / $env:DOCS_PORT
                                 override default ports (auto-fallback if busy)

  NOTE (auth): Tiled anonymous access is READ-ONLY; writes (ingest) require the
  API key. No key is hardcoded - this script GENERATES a strong TILED_API_KEY into
  backend\.env (gitignored) on first run, passes it to Tiled via --api-key, and the
  backend resolves the same value server-side. Servers bind to 127.0.0.1 only.
#>

# Orchestration script: handle errors explicitly (check $LASTEXITCODE for external
# tools, try/catch for cmdlets) rather than aborting on the first non-terminating error.
$ErrorActionPreference = 'Continue'

# ---------------------------------------------------------------------------
# Paths - the script lives in <repo>\windows, so the repo root is its parent.
# ---------------------------------------------------------------------------
$ScriptDir    = $PSScriptRoot
$RepoRoot     = Split-Path -Parent $ScriptDir
$BackendDir   = Join-Path $RepoRoot 'backend'
$FrontendDir  = Join-Path $RepoRoot 'frontend'
$TiledConfig  = Join-Path $RepoRoot 'tiled\config.yml'
$MkdocsConfig = Join-Path $RepoRoot 'mkdocs.yml'
$RunDir       = Join-Path $RepoRoot '.run'
$StaticDir    = Join-Path $BackendDir 'static'
$EnvDir       = Join-Path $RepoRoot '.venv'
$Python       = Join-Path $EnvDir 'Scripts\python.exe'

$TiledPidFile    = Join-Path $RunDir 'tiled.pid'
$BackendPidFile  = Join-Path $RunDir 'backend.pid'
$FrontendPidFile = Join-Path $RunDir 'frontend.pid'
$DocsPidFile     = Join-Path $RunDir 'docs.pid'

$ReqPyMajor = 3
$ReqPyMinor = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host $m -ForegroundColor Green }
function Warn($m) { Write-Host $m -ForegroundColor Yellow }
function Err($m)  { Write-Host $m -ForegroundColor Red }

function EnvOr([string]$Name, [string]$Default) {
  $v = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($v)) { return $Default }
  return $v
}

# True if something is already listening on 127.0.0.1:$Port. Uses a short TCP
# connect probe (no cmdlet dependency - mirrors the bash Python-socket check).
function Test-PortListening([int]$Port) {
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
    $done = $iar.AsyncWaitHandle.WaitOne(200)
    return ($done -and $client.Connected)
  } catch {
    return $false
  } finally {
    $client.Close()
  }
}

# Echo the first free port at/above $Start (scanning up to +50), or exit if none.
function Get-FreePort([int]$Start, [string]$Label) {
  for ($p = $Start; $p -le ($Start + 50); $p++) {
    if (-not (Test-PortListening $p)) { return $p }
  }
  Err "Error: no free $Label port in $Start..$($Start + 50) on 127.0.0.1."
  exit 1
}

# Kill a process and its whole child tree (taskkill /T reaps node/python grandchildren
# that Stop-Process alone would orphan).
function Stop-ProcessTree([int]$ProcId) {
  if (-not $ProcId) { return }
  & taskkill /PID $ProcId /T /F 2>$null | Out-Null
}

function Remove-PidFile([string]$Path) {
  if (Test-Path $Path) { Remove-Item $Path -Force -ErrorAction SilentlyContinue }
}

function Stop-ManagedProcess([string]$PidFile, [string]$Label) {
  if (-not (Test-Path $PidFile)) { return }
  $procId = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($procId -match '^\d+$') {
    if (Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue) {
      Warn "    Stopping stale $Label process from previous run (PID $procId)"
      Stop-ProcessTree ([int]$procId)
    }
  }
  Remove-PidFile $PidFile
}

# PIDs listening on a port (best-effort; empty if Get-NetTCPConnection unavailable).
function Get-ListenerPids([int]$Port) {
  try {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $conns) { return @() }
    return @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
  } catch {
    return @()
  }
}

# Reclaim a default port only from OUR OWN stale process: the listener's command
# line must reference this repo AND match the service pattern(s). A foreign process
# on the port is left alone (we fall back to the next free port).
function Stop-RepoListenerOnPort([int]$Port, [string]$Label, [string[]]$Patterns) {
  if (-not (Test-PortListening $Port)) { return }
  foreach ($procId in (Get-ListenerPids $Port)) {
    if (-not $procId) { continue }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
    if (-not $proc -or -not $proc.CommandLine) { continue }
    $cmd = $proc.CommandLine
    if ($cmd -notlike "*$RepoRoot*") { continue }
    $patMatch = $true
    foreach ($pat in $Patterns) { if ($cmd -notlike "*$pat*") { $patMatch = $false; break } }
    if ($patMatch) {
      Warn "    Reclaiming $Label port $Port from stale repo process (PID $procId)"
      Stop-ProcessTree ([int]$procId)
    }
  }
}

# Poll an HTTP endpoint until it answers with an acceptable status (or the backing
# process dies, or we run out of retries).
function Wait-HttpReady([string]$Url, [int]$Retries, [int[]]$OkCodes, $Proc) {
  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
      if ($OkCodes -contains [int]$resp.StatusCode) { return $true }
    } catch {
      $code = $null
      if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
      if ($code -and ($OkCodes -contains $code)) { return $true }
    }
    if ($Proc -and $Proc.HasExited) { return $false }
    Start-Sleep -Milliseconds 500
  }
  return $false
}

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
function Ensure-Uv {
  if (Get-Command uv -ErrorAction SilentlyContinue) { return }
  Warn "    uv not found - installing via astral.sh..."
  try {
    & powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  } catch {
    Err "Error: uv installation failed. Install manually and retry:"
    Err '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
  }
  # The installer drops uv in %USERPROFILE%\.local\bin - put it on PATH for this session.
  $uvBin = Join-Path $env:USERPROFILE '.local\bin'
  if (Test-Path $uvBin) { $env:PATH = "$uvBin;$env:PATH" }
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Err "Error: uv installed but not on PATH. Open a new PowerShell window and re-run windows\start_all.ps1."
    exit 1
  }
  Ok "    uv installed."
}

function Ensure-BackendEnv {
  Ensure-Uv

  if (-not (Test-Path $Python)) {
    Warn "    Creating .venv with Python $ReqPyMajor.$ReqPyMinor via uv..."
    & uv venv --python "$ReqPyMajor.$ReqPyMinor" $EnvDir
    if ($LASTEXITCODE -ne 0) { Err "Error: 'uv venv' failed."; exit 1 }
  }

  $env:PATH = "$(Join-Path $EnvDir 'Scripts');$env:PATH"

  & $Python -c "import tiled, uvicorn" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Warn "    Installing backend dependencies via uv..."
    # Mirrors ../start_all.sh. python-multipart is required by FastAPI for the
    # /api/ingest/upload multipart route (also declared in backend/pyproject.toml).
    & uv pip install --python $Python `
      "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" "tiled[all]>=0.1" `
      "numpy>=1.26" "pillow>=10.3" "python-dotenv>=1.0" "matplotlib>=3.8" `
      "pycocotools>=2.0.7" "scikit-image>=0.22" "tifffile>=2024.0" "imagecodecs"
    if ($LASTEXITCODE -ne 0) { Err "Error: backend dependency install failed."; exit 1 }
  }
}

function Ensure-FrontendRuntime {
  if (Get-Command npm -ErrorAction SilentlyContinue) { return }
  Err "Error: npm / Node.js 18+ was not found on PATH."
  Err "Install Node.js from https://nodejs.org or via winget (winget install OpenJS.NodeJS), then retry."
  exit 1
}

# Best-effort: never abort startup - a failure just means the in-app Docs link won't
# resolve. Returns $true if mkdocs is available.
function Ensure-DocsEnv {
  $mkdocs = Join-Path $EnvDir 'Scripts\mkdocs.exe'
  & $Python -c "import material" 2>$null
  if ((Test-Path $mkdocs) -and ($LASTEXITCODE -eq 0)) { return $true }
  Warn "    Installing docs dependencies (mkdocs-material) via uv..."
  try {
    & uv pip install --python $Python -q -r (Join-Path $RepoRoot 'docs\requirements.txt') 2>$null | Out-Null
  } catch { }
  & $Python -c "import material" 2>$null
  return ((Test-Path $mkdocs) -and ($LASTEXITCODE -eq 0))
}

# ---------------------------------------------------------------------------
# .env - create from .env.example if missing, then load into this session so
# child processes inherit it.
# ---------------------------------------------------------------------------
function Import-DotEnv {
  $envFile = Join-Path $BackendDir '.env'
  if (-not (Test-Path $envFile)) {
    Warn "    No .env found - copying from .env.example"
    Copy-Item (Join-Path $BackendDir '.env.example') $envFile
  }
  foreach ($line in (Get-Content $envFile)) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    $idx = $t.IndexOf('=')
    if ($idx -lt 1) { continue }
    $name = $t.Substring(0, $idx).Trim()
    $val  = $t.Substring($idx + 1).Trim()
    if ($val.Length -ge 2) {
      $q = $val[0]
      if (($q -eq '"' -or $q -eq "'") -and $val[$val.Length - 1] -eq $q) {
        $val = $val.Substring(1, $val.Length - 2)
      }
    }
    Set-Item -Path "Env:$name" -Value $val
  }
}

function New-TiledKey {
  $bytes = New-Object 'System.Byte[]' 16
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  return ([Convert]::ToBase64String($bytes) -replace '[^A-Za-z0-9]', '')
}

# Generate a strong Tiled API key when it's blank (fresh install) or still the old
# committed/leaked value (auto-rotate it). Persisted to backend\.env and exported.
function Ensure-TiledKey {
  $leaked = '3b1d23cdd45e7ada521c729cbd71763dd51b058e0a3e0c1cdeddbcbe13168c88'
  $current = $env:TILED_API_KEY
  if (-not ([string]::IsNullOrWhiteSpace($current)) -and $current -ne $leaked) { return }

  $key = New-TiledKey
  $envFile = Join-Path $BackendDir '.env'
  $lines = @()
  if (Test-Path $envFile) { $lines = @(Get-Content $envFile) }
  $found = $false
  $out = foreach ($ln in $lines) {
    if ($ln.Trim().StartsWith('TILED_API_KEY=')) { $found = $true; "TILED_API_KEY=$key" }
    else { $ln }
  }
  $out = @($out)
  if (-not $found) { $out += "TILED_API_KEY=$key" }
  # Write UTF-8 without BOM so python-dotenv reads the first line cleanly.
  [System.IO.File]::WriteAllLines($envFile, $out, (New-Object System.Text.UTF8Encoding($false)))
  $env:TILED_API_KEY = $key
  Ok "    Generated a new Tiled API key -> backend\.env"
}

# ---------------------------------------------------------------------------
# SAM (Magic tool) model - vendor SlimSAM locally so "Smart (AI)" works offline.
# Best-effort + backgrounded: never blocks or fails startup.
# ---------------------------------------------------------------------------
function Start-SamVendor {
  $modelDir = Join-Path $FrontendDir 'public\models\slimsam-77-uniform'
  if ((Test-Path $modelDir) -and (Get-ChildItem $modelDir -ErrorAction SilentlyContinue)) {
    Ok "    SAM model already vendored."
    return $null
  }
  Info "==> Vendoring SlimSAM model for the Magic tool (background)..."
  return Start-Job -ScriptBlock {
    param($PyExe, $ModelDir)
    & $PyExe -c "import huggingface_hub" 2>$null
    if ($LASTEXITCODE -ne 0) { & uv pip install --python $PyExe -q huggingface_hub 2>$null }
    & $PyExe -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Xenova/slimsam-77-uniform', local_dir=sys.argv[1])" $ModelDir
  } -ArgumentList $Python, $modelDir
}

# ===========================================================================
# Main
# ===========================================================================
$TiledPort    = [int](EnvOr 'TILED_PORT' '8010')
$BackendPort  = [int](EnvOr 'BACKEND_PORT' '8002')
$FrontendPort = [int](EnvOr 'FRONTEND_PORT' '5173')
$DocsPort     = [int](EnvOr 'DOCS_PORT' '8000')
$Prod = ((EnvOr 'PROD' '0') -eq '1') -or ((EnvOr 'SERVE_MODE' '') -eq 'prod')
$FrontendMode = if ($Prod) { 'prod' } else { 'dev' }

Ensure-BackendEnv
Ensure-FrontendRuntime

# Stop stale processes we tracked last run, then reclaim our own orphaned ports.
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Stop-ManagedProcess $DocsPidFile 'docs'
Stop-ManagedProcess $FrontendPidFile 'frontend'
Stop-ManagedProcess $BackendPidFile 'backend'
Stop-ManagedProcess $TiledPidFile 'Tiled'
Stop-RepoListenerOnPort $FrontendPort 'frontend' @('vite')
Stop-RepoListenerOnPort $BackendPort  'backend'  @('annotation_server:app')
Stop-RepoListenerOnPort $DocsPort     'docs'     @('mkdocs')
Stop-RepoListenerOnPort $TiledPort    'Tiled'    @($TiledConfig)

# Port fallback: if a default is taken by something we don't manage, use the next free one.
$orig = $TiledPort;    $TiledPort    = Get-FreePort $TiledPort 'Tiled'
if ($TiledPort -ne $orig)    { Warn "    Tiled port $orig is in use - using $TiledPort instead." }
$orig = $BackendPort;  $BackendPort  = Get-FreePort $BackendPort 'Backend'
if ($BackendPort -ne $orig)  { Warn "    Backend port $orig is in use - using $BackendPort instead." }
$orig = $FrontendPort; $FrontendPort = Get-FreePort $FrontendPort 'Frontend'
if ($FrontendPort -ne $orig) { Warn "    Frontend port $orig is in use - using $FrontendPort instead." }
$orig = $DocsPort;     $DocsPort     = Get-FreePort $DocsPort 'Docs'
if ($DocsPort -ne $orig)     { Warn "    Docs port $orig is in use - using $DocsPort instead." }

# Load .env, THEN export resolved-port vars so they win over any values baked into .env.
Import-DotEnv
$env:API_PROXY_TARGET = "http://127.0.0.1:$BackendPort"   # Vite dev proxy target
$env:VITE_DOCS_URL    = "http://127.0.0.1:$DocsPort"      # in-app Docs link
$env:TILED_URI        = "http://127.0.0.1:$TiledPort"     # backend -> Tiled

Ensure-TiledKey
# Defensive: a blank key would build an invalid "Authorization: Apikey " header.
if ([string]::IsNullOrWhiteSpace($env:TILED_API_KEY))       { Remove-Item Env:TILED_API_KEY -ErrorAction SilentlyContinue }
if ([string]::IsNullOrWhiteSpace($env:TILED_LOCAL_API_KEY)) { Remove-Item Env:TILED_LOCAL_API_KEY -ErrorAction SilentlyContinue }

$samJob = Start-SamVendor

# Services we launch + supervise. Each entry: Proc (Process), PidFile, Label, Required.
$managed = @()
$docsProc = $null
# Expected shutdown reasons (thrown after we've already logged context) - swallowed so
# the finally cleanup isn't followed by a noisy unhandled-error dump.
$sentinels = @(
  'missing-tiled-config', 'tiled-init-failed', 'tiled-not-ready', 'frontend-build-failed',
  'uvicorn-missing', 'backend-not-ready', 'service-exited'
)

try {
  # -------------------------------------------------------------------------
  # Tiled
  # -------------------------------------------------------------------------
  Info "==> Starting Tiled (port $TiledPort)..."
  & $Python (Join-Path $BackendDir 'scripts\repair_catalog_paths.py')
  if ($LASTEXITCODE -ne 0) { Warn "    (catalog path repair reported an issue - continuing)" }

  if (-not (Test-Path $TiledConfig)) { Err "Error: missing $TiledConfig"; throw 'missing-tiled-config' }

  $tiledDataDir = Join-Path $RepoRoot '.tiled'
  New-Item -ItemType Directory -Force -Path $tiledDataDir | Out-Null
  $tiledExe = Join-Path $EnvDir 'Scripts\tiled.exe'
  if (-not (Test-Path (Join-Path $tiledDataDir 'catalog.db'))) {
    Warn "    Initializing Tiled catalog (first run)..."
    Push-Location $RepoRoot
    & $tiledExe catalog init --if-not-exists "sqlite+aiosqlite:///./.tiled/catalog.db"
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Err "    Tiled catalog init failed. Ensure tiled[all] installed."; throw 'tiled-init-failed' }
  }

  # Pass the key via --api-key so Tiled's single_user_api_key exactly matches what the
  # backend sends - robust across Tiled versions. The config path is quoted so a repo
  # under a path with spaces still works (Start-Process does not auto-quote arg items).
  $tiledArgs = @('serve', 'config', "`"$TiledConfig`"", '--host', '127.0.0.1', '--port', "$TiledPort")
  if (-not [string]::IsNullOrWhiteSpace($env:TILED_API_KEY)) { $tiledArgs += @('--api-key', $env:TILED_API_KEY) }
  $tiledProc = Start-Process -FilePath $tiledExe -ArgumentList $tiledArgs -WorkingDirectory $RepoRoot -PassThru -NoNewWindow
  Set-Content -Path $TiledPidFile -Value $tiledProc.Id
  Ok "    Tiled PID: $($tiledProc.Id)"
  $managed += @{ Proc = $tiledProc; PidFile = $TiledPidFile; Label = 'Tiled'; Required = $true }

  Info "    Waiting for Tiled..."
  if (Wait-HttpReady "http://127.0.0.1:$TiledPort/" 40 @(200, 301, 302, 401, 403, 404) $tiledProc) {
    Ok "    Tiled ready at http://127.0.0.1:$TiledPort"
  } else {
    Err "    Tiled did not become ready in time."
    throw 'tiled-not-ready'
  }

  # -------------------------------------------------------------------------
  # Production SPA build (PROD=1): build + stage in backend\static BEFORE the
  # backend starts (the SPA mount is decided at import time). Dev: remove any
  # stale build so the backend stays API-only and Vite owns the SPA.
  # -------------------------------------------------------------------------
  if ($FrontendMode -eq 'prod') {
    Info "==> Building production frontend..."
    Push-Location $FrontendDir
    if (-not (Test-Path 'node_modules')) { Warn "    node_modules not found - running npm install..."; & npm install }
    & npm run build
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Err "    Frontend build failed."; throw 'frontend-build-failed' }
    if (Test-Path $StaticDir) { Remove-Item $StaticDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $StaticDir | Out-Null
    Copy-Item (Join-Path $FrontendDir 'dist\*') $StaticDir -Recurse -Force
    Ok "    Built SPA -> backend\static (served by the backend at :$BackendPort)."
  } elseif (Test-Path $StaticDir) {
    Remove-Item $StaticDir -Recurse -Force
  }

  # -------------------------------------------------------------------------
  # Backend
  # -------------------------------------------------------------------------
  Info "==> Starting backend (port $BackendPort)..."
  $uvicornExe = Join-Path $EnvDir 'Scripts\uvicorn.exe'
  if (-not (Test-Path $uvicornExe)) { Err "Error: uvicorn not found in $EnvDir."; throw 'uvicorn-missing' }
  $backendProc = Start-Process -FilePath $uvicornExe `
    -ArgumentList @('annotation_server:app', '--host', '127.0.0.1', '--port', "$BackendPort") `
    -WorkingDirectory $BackendDir -PassThru -NoNewWindow
  Set-Content -Path $BackendPidFile -Value $backendProc.Id
  Ok "    Backend PID: $($backendProc.Id)"
  $managed += @{ Proc = $backendProc; PidFile = $BackendPidFile; Label = 'backend'; Required = $true }

  Info "    Waiting for backend..."
  if (Wait-HttpReady "http://127.0.0.1:$BackendPort/health" 20 @(200) $backendProc) {
    Ok "    Backend ready at http://127.0.0.1:$BackendPort"
  } else {
    Err "    Backend failed to start. Check the logs above."
    throw 'backend-not-ready'
  }

  # -------------------------------------------------------------------------
  # Docs (best-effort)
  # -------------------------------------------------------------------------
  if ((Test-Path $MkdocsConfig) -and (Ensure-DocsEnv)) {
    Info "==> Starting docs (port $DocsPort)..."
    $mkdocsExe = Join-Path $EnvDir 'Scripts\mkdocs.exe'
    $docsProc = Start-Process -FilePath $mkdocsExe `
      -ArgumentList @('serve', '-f', "`"$MkdocsConfig`"", '-a', "127.0.0.1:$DocsPort") `
      -WorkingDirectory $RepoRoot -PassThru -NoNewWindow
    Set-Content -Path $DocsPidFile -Value $docsProc.Id
    Ok "    Docs PID: $($docsProc.Id)"
    $managed += @{ Proc = $docsProc; PidFile = $DocsPidFile; Label = 'docs'; Required = $false }
  } else {
    Warn "==> Skipping docs (mkdocs-material unavailable) - the in-app Docs link may not resolve."
  }

  # -------------------------------------------------------------------------
  # Frontend (dev only - in prod the backend already serves the built SPA)
  # -------------------------------------------------------------------------
  if ($FrontendMode -eq 'prod') {
    Info "==> Frontend served by the backend (prod build) at http://127.0.0.1:$BackendPort"
  } else {
    Info "==> Starting frontend (port $FrontendPort)..."
    Push-Location $FrontendDir
    if (-not (Test-Path 'node_modules')) { Warn "    node_modules not found - running npm install..."; & npm install }
    Pop-Location
    # npm is a .cmd batch file; Start-Process -NoNewWindow uses CreateProcess, which can't
    # launch a batch file directly ("%1 is not a valid Win32 application"). Run it via cmd.exe.
    $frontendProc = Start-Process -FilePath $env:ComSpec `
      -ArgumentList @('/c', 'npm', 'run', 'dev', '--', '--host', '--port', "$FrontendPort") `
      -WorkingDirectory $FrontendDir -PassThru -NoNewWindow
    Set-Content -Path $FrontendPidFile -Value $frontendProc.Id
    Ok "    Frontend PID: $($frontendProc.Id)"
    $managed += @{ Proc = $frontendProc; PidFile = $FrontendPidFile; Label = 'frontend'; Required = $true }
  }

  # -------------------------------------------------------------------------
  # Summary
  # -------------------------------------------------------------------------
  Write-Host ""
  Ok "=========================================="
  Ok "  Segmentation Annotation Studio is running!"
  Ok "  Tiled    : http://127.0.0.1:$TiledPort (public / anonymous)"
  if ($FrontendMode -eq 'prod') {
    Ok "  App (prod build) : http://127.0.0.1:$BackendPort"
  } else {
    Ok "  Frontend : http://127.0.0.1:$FrontendPort"
  }
  Ok "  Backend  : http://127.0.0.1:$BackendPort"
  if ($docsProc) { Ok "  Docs     : http://127.0.0.1:$DocsPort" }
  Ok "  Press Ctrl+C to stop all servers."
  Ok "=========================================="
  Write-Host ""

  # Supervise: block until Ctrl+C, or exit if a required service dies.
  while ($true) {
    Start-Sleep -Seconds 1
    foreach ($m in $managed) {
      if ($m.Required -and $m.Proc -and $m.Proc.HasExited) {
        Err "    $($m.Label) exited unexpectedly (code $($m.Proc.ExitCode)). Shutting down."
        throw 'service-exited'
      }
    }
  }
} catch {
  $msg = "$($_.Exception.Message)"
  if ($sentinels -notcontains $msg) { Err "    Unexpected error: $msg" }
} finally {
  Write-Host ""
  Warn "Shutting down..."
  foreach ($m in $managed) {
    if ($m.Proc) { Stop-ProcessTree $m.Proc.Id }
    if ($m.PidFile) { Remove-PidFile $m.PidFile }
  }
  if ($samJob) {
    Stop-Job $samJob -ErrorAction SilentlyContinue
    Remove-Job $samJob -Force -ErrorAction SilentlyContinue
  }
  Ok "Done."
}
