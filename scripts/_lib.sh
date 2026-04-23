#!/usr/bin/env bash
# Shared helpers for spark-lab pipeline. Source from any phase script.
# Strict mode + traceback + structured logging.

set -Eeuo pipefail

SPARK_LAB_ROOT="${SPARK_LAB_ROOT:-$HOME/spark-lab}"
SPARK_LAB_LOG="${SPARK_LAB_LOG:-$SPARK_LAB_ROOT/logs/pipeline.log}"
SPARK_LAB_EVENTS="${SPARK_LAB_EVENTS:-$SPARK_LAB_ROOT/events.log}"
SPARK_LAB_HEARTBEAT="${SPARK_LAB_HEARTBEAT:-$SPARK_LAB_ROOT/heartbeat.txt}"
SPARK_LAB_LOCK="${SPARK_LAB_LOCK:-$SPARK_LAB_ROOT/.lock}"
SPARK_LAB_SERVICE_STATE="${SPARK_LAB_SERVICE_STATE:-$SPARK_LAB_ROOT/state/services_snapshot.json}"

# ---- logging ----------------------------------------------------------------

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() {
  local lvl="$1"; shift
  local line
  line="$(ts) [$lvl] $*"
  echo "$line"
  echo "$line" >> "$SPARK_LAB_LOG"
}
info()  { log "INFO"  "$@"; }
warn()  { log "WARN"  "$@"; }
error() { log "ERROR" "$@"; }
event() {
  local kind="$1"; shift
  echo "$(ts) $kind $*" >> "$SPARK_LAB_EVENTS"
  log "EVENT" "$kind $*"
}

heartbeat() {
  local stage="$1"; shift || true
  printf '%s alive stage=%s %s\n' "$(ts)" "$stage" "${*:-}" > "$SPARK_LAB_HEARTBEAT"
}

# Single-line failure handler — fired on any non-zero exit via trap
on_error() {
  local code=$?
  error "phase failed exit=$code at line $1: $BASH_COMMAND"
  event "PHASE_FAIL" "exit=$code line=$1 cmd=$BASH_COMMAND"
  return $code
}
trap 'on_error $LINENO' ERR

# ---- lock -------------------------------------------------------------------

acquire_lock() {
  if [ -e "$SPARK_LAB_LOCK" ]; then
    local owner
    owner=$(cat "$SPARK_LAB_LOCK" 2>/dev/null || echo "?")
    if kill -0 "${owner%% *}" 2>/dev/null; then
      error "pipeline already running (lock held by pid ${owner%% *})"
      return 1
    else
      warn "stale lock from pid ${owner%% *}, claiming"
      rm -f "$SPARK_LAB_LOCK"
    fi
  fi
  echo "$$ $(ts)" > "$SPARK_LAB_LOCK"
  trap 'release_lock' EXIT
  info "lock acquired pid=$$"
}

release_lock() {
  if [ -e "$SPARK_LAB_LOCK" ]; then
    local owner
    owner=$(awk '{print $1}' "$SPARK_LAB_LOCK")
    if [ "$owner" = "$$" ]; then
      rm -f "$SPARK_LAB_LOCK"
      info "lock released pid=$$"
    fi
  fi
}

# ---- preflight gates --------------------------------------------------------

# Abort if free disk on $1 (default $HOME) below $2 GB (default 30)
require_disk_gb() {
  local path="${1:-$HOME}"
  local need="${2:-30}"
  local avail_gb
  avail_gb=$(df -BG --output=avail "$path" | tail -1 | tr -dc '0-9')
  if [ "$avail_gb" -lt "$need" ]; then
    error "disk too low on $path: ${avail_gb}G free, need ${need}G"
    return 1
  fi
  info "disk ok on $path: ${avail_gb}G free (need ${need}G)"
}

# Require a HF_TOKEN env var to be set (and non-empty)
require_hf_token() {
  if [ -z "${HF_TOKEN:-}" ]; then
    error "HF_TOKEN env var not set"
    return 1
  fi
  info "HF_TOKEN present"
}

# ---- service preservation ---------------------------------------------------

# Capture currently-running long-lived inference services so we can restore
# them after a heavy quant/eval phase. Snapshot is JSON written to
# $SPARK_LAB_SERVICE_STATE.
#
# We match by EXE name (/proc/<pid>/comm) + port presence in cmdline, NOT just
# substring match on cmdline — otherwise bash launcher wrappers that contain
# all the service strings get falsely matched.
snapshot_services() {
  local out="$SPARK_LAB_SERVICE_STATE"
  python3 - "$out" <<'PYEOF'
import json, os, subprocess, sys
# (label, expected /proc/<pid>/comm starts-with, port the process must bind)
SERVICES = [
    ("llama-server-8081", "llama-server", 8081),
    ("llama-server-8082", "llama-server", 8082),
    ("kokoro-8880",       "uvicorn",      8880),
    ("whisper-shim-8765", "uvicorn",      8765),
    ("whisper-cpp-8766",  "whisper-serve", 8766),  # comm is truncated to 15 chars
]
def cmdline_of(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode(errors="ignore").strip()
    except FileNotFoundError:
        return ""
def comm_of(pid):
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
def cwd_of(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError):
        return None

out = sys.argv[1]
snap = []
# Get all PIDs listening on each port via ss
ss_out = subprocess.check_output(["ss", "-tlnp"], text=True)
port_to_pid = {}
for line in ss_out.splitlines():
    if "pid=" not in line:
        continue
    # parse "0.0.0.0:8081" and "pid=224553"
    addr_part = line.split()[3] if len(line.split()) > 3 else ""
    port = addr_part.rsplit(":", 1)[-1] if ":" in addr_part else ""
    pid_match = [t for t in line.split() if "pid=" in t]
    if not pid_match or not port.isdigit():
        continue
    # users:(("name",pid=NNNNN,fd=N))
    import re
    m = re.search(r"pid=(\d+)", line)
    if m:
        port_to_pid[int(port)] = int(m.group(1))

for label, expected_comm, port in SERVICES:
    pid = port_to_pid.get(port)
    if not pid:
        print(f"  warn: no listener on :{port} for {label}")
        continue
    comm = comm_of(pid)
    if not comm.startswith(expected_comm[:15]):
        print(f"  warn: :{port} pid={pid} comm={comm!r} doesn't match {expected_comm!r}")
        continue
    snap.append({
        "name": label, "pid": pid, "port": port,
        "comm": comm, "cmd": cmdline_of(pid), "cwd": cwd_of(pid),
    })

with open(out, "w") as f:
    json.dump(snap, f, indent=2)
print(f"snapshotted {len(snap)} services to {out}")
PYEOF
  info "service snapshot written: $(jq 'length' "$out" 2>/dev/null || echo '?') entries"
}

# Stop the snapshotted services (graceful TERM, fall back to KILL after 5s)
stop_services() {
  if [ ! -s "$SPARK_LAB_SERVICE_STATE" ]; then
    warn "no service snapshot to stop"
    return 0
  fi
  local pids
  pids=$(jq -r '.[].pid' "$SPARK_LAB_SERVICE_STATE")
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null && info "TERM sent to pid=$pid"
    fi
  done
  sleep 5
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null && warn "KILL sent to pid=$pid (TERM ignored)"
    fi
  done
}

# Restart services from snapshot — same cwd, same cmdline
restore_services() {
  if [ ! -s "$SPARK_LAB_SERVICE_STATE" ]; then
    warn "no service snapshot to restore"
    return 0
  fi
  python3 - "$SPARK_LAB_SERVICE_STATE" <<'PYEOF'
import json, os, subprocess, sys, time
snap = json.load(open(sys.argv[1]))
for s in snap:
    cwd = s.get("cwd") or os.path.expanduser("~")
    cmd = s["cmd"]
    name = s["name"]
    log = f"/tmp/{name}.restored.log"
    full = f"cd {cwd!r} && nohup {cmd} > {log} 2>&1 &"
    print(f"restoring {name}: {full}")
    subprocess.Popen(["bash", "-c", full + " disown"], start_new_session=True)
time.sleep(3)
print("restore commands dispatched")
PYEOF
  info "services restore dispatched (verify ports yourself)"
}

# ---- helpers ----------------------------------------------------------------

# Atomically write content to a file via a tempfile rename
atomic_write() {
  local target="$1"; local content="$2"
  local tmp; tmp="$(mktemp "${target}.XXXX")"
  printf '%s' "$content" > "$tmp"
  mv "$tmp" "$target"
}

# Run a command with a wall-clock budget (kill if exceeds)
with_timeout() {
  local secs="$1"; shift
  timeout --signal=TERM --kill-after=30s "${secs}s" "$@"
}
