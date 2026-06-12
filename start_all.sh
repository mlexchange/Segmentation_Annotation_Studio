#!/usr/bin/env bash
# start_all.sh — Start Tiled, Browse backend, and frontend together.
# Usage: ./start_all.sh
# Stop everything: Ctrl+C

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
  if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}Error: uv is not installed.${NC}"
    echo -e "${RED}Install with: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    echo -e "${RED}Then open a new shell (or run: source \$HOME/.local/bin/env) and retry.${NC}"
    exit 1
  fi
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
require_free_port "$TILED_PORT" "Tiled"
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

# Generate a stable API key if TILED_API_KEY is empty or missing
if [ -z "${TILED_API_KEY:-}" ]; then
  TILED_API_KEY=$("$PYTHON" -c "import secrets; print(secrets.token_hex(32))")
  # Persist it back into .env so it survives restarts
  if grep -q "^TILED_API_KEY=" "$BACKEND_DIR/.env"; then
    sed -i.bak "s|^TILED_API_KEY=.*|TILED_API_KEY=${TILED_API_KEY}|" "$BACKEND_DIR/.env" && rm -f "$BACKEND_DIR/.env.bak"
  else
    echo "TILED_API_KEY=${TILED_API_KEY}" >> "$BACKEND_DIR/.env"
  fi
  echo -e "${YELLOW}    Generated new TILED_API_KEY and saved to backend/.env${NC}"
fi
export TILED_API_KEY

# ---------------------------------------------------------------------------
# Tiled (must match backend/tiled_config.py default: port 8010)
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
  (cd "$SCRIPT_DIR" && TILED_SINGLE_USER_API_KEY="$TILED_API_KEY" tiled_cmd catalog init --if-not-exists \
    "sqlite+aiosqlite:///./.tiled/catalog.db") || {
    echo -e "${RED}    Tiled catalog init failed. Install: pip install 'tiled[server]'${NC}"
    exit 1
  }
fi

(cd "$SCRIPT_DIR" && TILED_SINGLE_USER_API_KEY="$TILED_API_KEY" tiled_cmd serve config "$TILED_CONFIG" --host 127.0.0.1 --port "$TILED_PORT") &
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
  if ! kill -0 "$TILED_PID" 2>/dev/null; then
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
echo -e "${GREEN}  Tiled    : http://127.0.0.1:${TILED_PORT}${NC}"
echo -e "${GREEN}  Frontend : http://127.0.0.1:${FRONTEND_PORT}${NC}"
echo -e "${GREEN}  Backend  : http://127.0.0.1:${BACKEND_PORT}${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop all servers.${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

wait
