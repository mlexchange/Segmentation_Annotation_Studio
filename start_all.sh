#!/usr/bin/env bash
# start_all.sh — Start Tiled, Browse backend, and frontend together.
# Usage: ./start_all.sh
# Stop everything: Ctrl+C
#
# NOTE: Tiled auth lives in tiled/config.yml:
#   authentication:
#     allow_anonymous_access: true      # anonymous access is READ-ONLY
#     single_user_api_key: "${TILED_API_KEY}"   # required for WRITES (e.g. ingest)
# No key is hardcoded: this script GENERATES a strong TILED_API_KEY into
# backend/.env (gitignored) on first run and exports it; tiled/config.yml pulls it
# via ${TILED_API_KEY}, and the backend resolves the same value server-side
# (backend/tiled_config.py) — never exposing it to the frontend.
# Server binds to 127.0.0.1 (local-only). Do NOT change --host to 0.0.0.0
# without reconsidering the auth posture.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
TILED_CONFIG="$SCRIPT_DIR/tiled/config.yml"
TILED_PORT="${TILED_PORT:-8010}"
BACKEND_PORT="${BACKEND_PORT:-8002}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
RUN_DIR="$SCRIPT_DIR/.run"
TILED_PID_FILE="$RUN_DIR/tiled.pid"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
ENV_DIR=""
ENV_KIND=""
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

NPM_CMD=()

port_is_listening() {
  local port="$1"
  "$PYTHON" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

require_free_port() {
  local port="$1"
  local label="$2"
  if port_is_listening "$port"; then
    echo -e "${RED}Error: ${label} port ${port} is already in use on 127.0.0.1.${NC}"
    echo -e "${RED}Stop the existing process or rerun with a different port.${NC}"
    exit 1
  fi
}

# Echo the first free port at/above $1 (scanning up to +50), or exit if none.
# Only the chosen port goes to stdout; status messages go to stderr.
pick_free_port() {
  local port="$1" label="$2" p="$1" max=$(( $1 + 50 ))
  while [ "$p" -le "$max" ]; do
    if ! port_is_listening "$p"; then
      echo "$p"
      return 0
    fi
    p=$(( p + 1 ))
  done
  echo -e "${RED}Error: no free ${label} port in ${port}..${max} on 127.0.0.1.${NC}" >&2
  exit 1
}

cleanup_pid_file() {
  local pid_file="$1"
  rm -f "$pid_file"
}

stop_managed_process() {
  local pid_file="$1"
  local label="$2"

  if [ ! -f "$pid_file" ]; then
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    cleanup_pid_file "$pid_file"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo -e "${YELLOW}    Stopping stale ${label} process from previous run (PID ${pid})${NC}"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  cleanup_pid_file "$pid_file"
}

cleanup_managed_processes() {
  mkdir -p "$RUN_DIR"
  stop_managed_process "$FRONTEND_PID_FILE" "frontend"
  stop_managed_process "$BACKEND_PID_FILE" "backend"
  stop_managed_process "$TILED_PID_FILE" "Tiled"
}

get_process_command() {
  local pid="$1"
  ps -o command= -p "$pid" 2>/dev/null | sed 's/^[[:space:]]*//'
}

get_process_cwd() {
  local pid="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | while IFS= read -r line; do
    [[ "$line" == n* ]] || continue
    printf '%s\n' "${line#n}"
    break
  done
}

find_listener_processes() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  lsof -nP -iTCP:"$port" -sTCP:LISTEN -Fp 2>/dev/null | while IFS= read -r line; do
    [[ "$line" == p* ]] || continue
    local pid="${line#p}"
    local cmdline=""
    local cwd=""
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmdline="$(get_process_command "$pid")"
    cwd="$(get_process_cwd "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$cmdline" "$cwd"
  done
}

stop_repo_listener_on_port() {
  local port="$1"
  local label="$2"
  local expected_dir="$3"
  local pattern_a="$4"
  local pattern_b="$5"
  local matches=""

  if ! port_is_listening "$port"; then
    return
  fi

  matches="$(find_listener_processes "$port")"
  if [ -z "$matches" ]; then
    return
  fi

  while IFS=$'\t' read -r pid cmdline cwd; do
    [ -n "$pid" ] || continue
    if ([[ -n "$expected_dir" ]] && [[ "$cwd" == "$expected_dir"* ]]) || [[ "$cmdline" == *"$SCRIPT_DIR"* ]]; then
      if ([[ -z "$pattern_a" ]] || [[ "$cmdline" == *"$pattern_a"* ]]) && ([[ -z "$pattern_b" ]] || [[ "$cmdline" == *"$pattern_b"* ]]); then
        echo -e "${YELLOW}    Reclaiming ${label} port ${port} from stale repo process (PID ${pid})${NC}"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
          if ! kill -0 "$pid" 2>/dev/null; then
            break
          fi
          sleep 0.2
        done
        if kill -0 "$pid" 2>/dev/null; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    elif [[ "$cmdline" == *"$pattern_a"* ]] && ([[ -z "$pattern_b" ]] || [[ "$cmdline" == *"$pattern_b"* ]]); then
      echo -e "${YELLOW}    Reclaiming ${label} port ${port} from stale repo process (PID ${pid})${NC}"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.2
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  done <<< "$matches"
}

reclaim_orphaned_repo_ports() {
  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi
  stop_repo_listener_on_port "$FRONTEND_PORT" "frontend" "$FRONTEND_DIR" "vite" ""
  stop_repo_listener_on_port "$BACKEND_PORT" "backend" "$BACKEND_DIR" "annotation_server:app" "uvicorn"
  stop_repo_listener_on_port "$TILED_PORT" "Tiled" "$SCRIPT_DIR" "$TILED_CONFIG" "tiled"
}

can_run_npm() {
  local npm_bin="$1"
  "$npm_bin" --version >/dev/null 2>&1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo -e "${YELLOW}    uv not found — installing via astral.sh...${NC}"
  if ! command -v curl >/dev/null 2>&1; then
    echo -e "${RED}Error: curl is required to auto-install uv. Install uv manually:${NC}"
    echo -e "${RED}  curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
  fi
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    echo -e "${RED}Error: uv installation failed. Install manually and retry.${NC}"
    exit 1
  fi
  # Put uv on PATH for this session (the installer drops it in ~/.local/bin).
  # shellcheck source=/dev/null
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}Error: uv installed but not on PATH. Open a new shell (or run:${NC}"
    echo -e "${RED}  source \$HOME/.local/bin/env) and re-run start_all.sh.${NC}"
    exit 1
  fi
  echo -e "${GREEN}    uv installed.${NC}"
}

ensure_backend_env() {
  ensure_uv

  ENV_DIR="$SCRIPT_DIR/.venv"
  ENV_KIND="venv"

  if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo -e "${YELLOW}    Creating .venv with Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} via uv...${NC}"
    uv venv --python "${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}" "$ENV_DIR"
  fi

  PYTHON="$ENV_DIR/bin/python"
  export PATH="$ENV_DIR/bin:$PATH"

  if ! "$PYTHON" -c "import tiled, uvicorn" >/dev/null 2>&1; then
    echo -e "${YELLOW}    Installing backend dependencies via uv...${NC}"
    uv pip install --python "$PYTHON" \
      "fastapi>=0.115" "uvicorn[standard]>=0.30" "tiled[all]>=0.1" \
      "numpy>=1.26" "pillow>=10.3" "python-dotenv>=1.0" "matplotlib>=3.8" \
      "pycocotools>=2.0.7" "scikit-image>=0.22" "tifffile>=2024.0" "imagecodecs"
  fi
}

ensure_frontend_runtime() {
  if command -v npm >/dev/null 2>&1 && can_run_npm "$(command -v npm)"; then
    NPM_CMD=("$(command -v npm)")
    return
  fi

  echo -e "${RED}Error: npm / Node.js 18+ was not found on PATH.${NC}"
  echo -e "${RED}Install Node.js from https://nodejs.org or via your package manager, then retry.${NC}"
  exit 1
}

tiled_cmd() {
  if [ -x "$ENV_DIR/bin/tiled" ]; then
    "$ENV_DIR/bin/tiled" "$@"
  elif command -v tiled &>/dev/null; then
    tiled "$@"
  else
    echo -e "${RED}Error: tiled CLI not found in $ENV_DIR or on PATH.${NC}"
    echo -e "${RED}Run the script again after environment bootstrap succeeds, or install dependencies into $ENV_DIR.${NC}"
    exit 1
  fi
}

cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down...${NC}"
  kill "$TILED_PID" "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$TILED_PID" "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  cleanup_pid_file "$TILED_PID_FILE"
  cleanup_pid_file "$BACKEND_PID_FILE"
  cleanup_pid_file "$FRONTEND_PID_FILE"
  echo -e "${GREEN}Done.${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

ensure_backend_env
ensure_frontend_runtime
cleanup_managed_processes
reclaim_orphaned_repo_ports
# Tiled: fall back to the next free port if the default is taken by something we
# don't manage (e.g. another app's Tiled). Point the backend at the chosen port.
_orig_tiled_port="$TILED_PORT"
TILED_PORT="$(pick_free_port "$TILED_PORT" "Tiled")"
if [ "$TILED_PORT" != "$_orig_tiled_port" ]; then
  echo -e "${YELLOW}    Tiled port ${_orig_tiled_port} is in use — using ${TILED_PORT} instead.${NC}"
fi
# Export so the backend (tiled_config.py) and its server list resolve the same
# local Tiled instance rather than the hardcoded default.
export TILED_URI="http://127.0.0.1:${TILED_PORT}"

require_free_port "$BACKEND_PORT" "Backend"
require_free_port "$FRONTEND_PORT" "Frontend"

# ---------------------------------------------------------------------------
# Load .env — create it from .env.example if missing
# ---------------------------------------------------------------------------
if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo -e "${YELLOW}    No .env found — copying from .env.example${NC}"
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
fi

set -a
# shellcheck source=/dev/null
source "$BACKEND_DIR/.env"
set +a

# Generate a strong Tiled API key so nothing is hardcoded. Runs when the key is
# blank (fresh install) OR still the old committed/leaked value (auto-rotate it).
# Persisted to backend/.env (gitignored) and exported so the Tiled server pulls it
# via ${TILED_API_KEY} in tiled/config.yml and the backend resolves the same value.
LEAKED_TILED_KEY="3b1d23cdd45e7ada521c729cbd71763dd51b058e0a3e0c1cdeddbcbe13168c88"
if [ -z "${TILED_API_KEY// }" ] || [ "$TILED_API_KEY" = "$LEAKED_TILED_KEY" ]; then
  # Alphanumeric only (Tiled validates single_user_api_key against [a-zA-Z0-9]+).
  NEW_KEY="$(openssl rand -base64 16 2>/dev/null | tr -dc 'A-Za-z0-9')"
  [ -z "$NEW_KEY" ] && NEW_KEY="$("$PYTHON" -c 'import secrets,base64,re;print(re.sub(r"[^A-Za-z0-9]","",base64.b64encode(secrets.token_bytes(16)).decode()))')"
  "$PYTHON" - "$BACKEND_DIR/.env" "$NEW_KEY" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); key = sys.argv[2]
lines = p.read_text().splitlines() if p.exists() else []
out, found = [], False
for ln in lines:
    out.append(f"TILED_API_KEY={key}" if ln.strip().startswith("TILED_API_KEY=") else ln)
    found = found or ln.strip().startswith("TILED_API_KEY=")
if not found:
    out.append(f"TILED_API_KEY={key}")
p.write_text("\n".join(out) + "\n")
PYEOF
  export TILED_API_KEY="$NEW_KEY"
  echo -e "${GREEN}    Generated a new Tiled API key → backend/.env${NC}"
fi

# Defensive: a blank TILED_* key exported here would make the Tiled client build an
# "Authorization: Apikey " (trailing space) header that httpx rejects. Unset any
# blank key vars (no-op once a key is generated above).
[ -z "${TILED_API_KEY// }" ] && unset TILED_API_KEY
[ -z "${TILED_LOCAL_API_KEY// }" ] && unset TILED_LOCAL_API_KEY

# ---------------------------------------------------------------------------
# SAM (Magic tool) model — vendor SlimSAM locally so "Smart (AI)" works offline.
# Best-effort + backgrounded: never blocks or fails startup (SAM falls back to
# the remote HF model if this can't complete).
# ---------------------------------------------------------------------------
ensure_sam_model() {
  local MODEL_DIR="$FRONTEND_DIR/public/models/slimsam-77-uniform"
  if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo -e "${GREEN}    SAM model already vendored.${NC}"
    return 0
  fi
  echo -e "${CYAN}==> Vendoring SlimSAM model for the Magic tool (background)...${NC}"
  (
    # The .venv is uv-managed and has no pip; install huggingface_hub via uv,
    # falling back to python -m pip only if uv is unavailable.
    if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
      if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON" -q huggingface_hub >/dev/null 2>&1 || true
      else
        "$PYTHON" -m pip install -q huggingface_hub >/dev/null 2>&1 || true
      fi
    fi
    if "$PYTHON" - "$MODEL_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Xenova/slimsam-77-uniform", local_dir=sys.argv[1])
print("SAM model vendored to", sys.argv[1])
PYEOF
    then
      echo -e "${GREEN}    SAM model vendored — hard-refresh the app to enable Smart (AI).${NC}"
    else
      echo -e "${YELLOW}    SAM model vendoring skipped (no huggingface_hub / offline) — Smart (AI) falls back to remote.${NC}"
    fi
  ) &
}
ensure_sam_model

# ---------------------------------------------------------------------------
# Tiled — local server. Anonymous access is READ-ONLY (allow_anonymous_access);
# writes (ingest) authenticate with tiled/config.yml's single_user_api_key,
# resolved server-side by backend/tiled_config.py. Keys are never sent to the
# frontend. (Port must match backend/tiled_config.py default: 8010.)
# ---------------------------------------------------------------------------
echo -e "${CYAN}==> Starting Tiled (port ${TILED_PORT})...${NC}"

# Repair catalog asset paths in case the repo was moved or cloned to a new location.
"$PYTHON" "$SCRIPT_DIR/backend/scripts/repair_catalog_paths.py"

if [ ! -f "$TILED_CONFIG" ]; then
  echo -e "${RED}Error: missing $TILED_CONFIG${NC}"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/.tiled"
if [ ! -f "$SCRIPT_DIR/.tiled/catalog.db" ]; then
  echo -e "${YELLOW}    Initializing Tiled catalog (first run)...${NC}"
  (cd "$SCRIPT_DIR" && tiled_cmd catalog init --if-not-exists \
    "sqlite+aiosqlite:///./.tiled/catalog.db") || {
    echo -e "${RED}    Tiled catalog init failed. Install: pip install 'tiled[server]'${NC}"
    exit 1
  }
fi

# tiled_cmd is a shell function (can't be exec'd); the subshell waits on the real
# tiled child. The ready-check below curls the port before declaring failure, so a
# dead wrapper alone won't trip a false "Tiled failed to start".
(cd "$SCRIPT_DIR" && tiled_cmd serve config "$TILED_CONFIG" --host 127.0.0.1 --port "$TILED_PORT") &
TILED_PID=$!
echo "$TILED_PID" > "$TILED_PID_FILE"
echo -e "${GREEN}    Tiled PID: $TILED_PID${NC}"

echo -e "${CYAN}    Waiting for Tiled...${NC}"
TILED_READY=0
for i in $(seq 1 40); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${TILED_PORT}/" 2>/dev/null || echo "000")
  if [[ "$code" =~ ^(200|301|302|401|403|404)$ ]]; then
    echo -e "${GREEN}    Tiled ready at http://127.0.0.1:${TILED_PORT}${NC}"
    TILED_READY=1
    break
  fi
  # Only treat a dead PID as failure if the port is ALSO not serving (avoids a
  # false negative from a wrapper exiting while Tiled itself is up).
  if ! kill -0 "$TILED_PID" 2>/dev/null; then
    code=$(curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${TILED_PORT}/" 2>/dev/null || echo "000")
    if [[ "$code" =~ ^(200|301|302|401|403|404)$ ]]; then
      echo -e "${GREEN}    Tiled ready at http://127.0.0.1:${TILED_PORT}${NC}"
      TILED_READY=1
      break
    fi
    echo -e "${RED}    Tiled failed to start. Install: pip install 'tiled[server]' (see backend/requirements.txt).${NC}"
    exit 1
  fi
  sleep 0.5
done
if [ "$TILED_READY" != 1 ]; then
  echo -e "${RED}    Tiled did not become ready in time (http code: ${code:-unknown}).${NC}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
echo -e "${CYAN}==> Starting backend (port ${BACKEND_PORT})...${NC}"

UVICORN_CMD=("$ENV_DIR/bin/uvicorn")
if [ ! -x "${UVICORN_CMD[0]}" ]; then
  echo -e "${RED}Error: uvicorn not found in $ENV_DIR.${NC}"
  echo -e "${RED}Run: $PYTHON -m pip install -r backend/requirements.txt${NC}"
  exit 1
fi

cd "$BACKEND_DIR"
"${UVICORN_CMD[@]}" annotation_server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
echo -e "${GREEN}    Backend PID: $BACKEND_PID${NC}"

echo -e "${CYAN}    Waiting for backend...${NC}"
for i in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo -e "${GREEN}    Backend ready at http://127.0.0.1:${BACKEND_PORT}${NC}"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo -e "${RED}    Backend failed to start. Check logs above.${NC}"
    exit 1
  fi
  sleep 0.5
done

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
echo -e "${CYAN}==> Starting frontend (port ${FRONTEND_PORT})...${NC}"

cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}    node_modules not found — running npm install...${NC}"
  "${NPM_CMD[@]}" install
fi

"${NPM_CMD[@]}" run dev -- --host --port "$FRONTEND_PORT" &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
echo -e "${GREEN}    Frontend PID: $FRONTEND_PID${NC}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}==========================================${NC}"
  echo -e "${GREEN}  SAM3 Annotation Studio is running!${NC}"
echo -e "${GREEN}  Tiled    : http://127.0.0.1:${TILED_PORT} (public / anonymous)${NC}"
echo -e "${GREEN}  Frontend : http://127.0.0.1:${FRONTEND_PORT}${NC}"
echo -e "${GREEN}  Backend  : http://127.0.0.1:${BACKEND_PORT}${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop all servers.${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

wait
