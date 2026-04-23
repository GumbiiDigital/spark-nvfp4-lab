#!/usr/bin/env bash
# Measure throughput against a running OpenAI-compatible endpoint at different
# concurrency levels. Outputs JSON with tok/s and latency stats.
#
# Usage: perf.sh <port> <out_file>
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

PORT="${1:?missing port}"
OUT_FILE="${2:?missing out_file}"

event "PERF_START" "port=$PORT out=$OUT_FILE"
heartbeat "perf:$PORT"

python3 - "$PORT" "$OUT_FILE" <<'PYEOF'
import json, statistics, sys, time, urllib.request

PORT = int(sys.argv[1]); OUT = sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPT = ("Write a single short paragraph about the history of the printing press. "
          "Be concise but factually correct.")
MAX_TOK = 256

def one_request():
    body = json.dumps({
        "model": "quant",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOK,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    completion_tokens = resp.get("usage", {}).get("completion_tokens") or MAX_TOK
    return dt, completion_tokens

def measure(concurrency, n_requests):
    import concurrent.futures
    times, toks = [], []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request) for _ in range(n_requests)]
        for f in concurrent.futures.as_completed(futs):
            try:
                dt, tk = f.result()
                times.append(dt); toks.append(tk)
            except Exception as e:
                print(f"req fail: {e}", file=sys.stderr)
    wall = time.perf_counter() - t0
    if not times:
        return {"concurrency": concurrency, "ok": False}
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_ok": len(times),
        "wall_s": round(wall, 3),
        "p50_latency_s": round(statistics.median(times), 3),
        "p95_latency_s": round(sorted(times)[int(len(times)*0.95)-1] if len(times) >= 20 else max(times), 3),
        "single_stream_tok_s": round(sum(toks) / sum(times), 2),
        "aggregate_tok_s": round(sum(toks) / wall, 2),
    }

# warmup
print("warmup...")
one_request(); one_request()

results = {"port": PORT, "max_tokens": MAX_TOK, "prompt_chars": len(PROMPT), "runs": []}
for c, n in [(1, 8), (4, 16), (16, 48)]:
    print(f"measuring concurrency={c} n_requests={n}")
    results["runs"].append(measure(c, n))

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}")
PYEOF

event "PERF_OK" "out=$OUT_FILE"
info "perf complete: $OUT_FILE"
