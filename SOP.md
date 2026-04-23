# Spark NVFP4 Lab — Release Pipeline SOP

Operational procedures for producing `release-N_<MODEL>` artifacts: NVFP4 quantization, apples-to-apples BF16 baseline, benchmark card.

Derived from the 2026-04-22/23 release-0 session (DeepSeek-R1-Distill-Llama-8B).

---

## 1. Layout

```
~/spark-lab/
  scripts/
    pipeline.sh         Orchestrator. State-machine over phases.
    quantize.sh         ModelOpt NVFP4 quant.
    serve.sh            Start/stop/wait trtllm-serve container.
    eval.sh             Run lm-eval via OpenAI-compatible API.
    perf.sh             Throughput/latency sweep.
    render_card.py      Produce README + BENCHMARKS + manifest + REPRODUCE.sh.
    _lib.sh             Shared helpers (logging, events, timeouts).
    run_humaneval.sh    Generation-task add-on (ships both models sequentially).
  logs/                 All per-phase logs land here. events.log is the event bus.
  results/<release>/    Per-release artifacts: quant/, eval-bf16/, eval-nvfp4/, perf/, state.json.
  models/               Materialized (de-symlinked) HF snapshots. See §4.
  heartbeat.txt         Last-known liveness tick. Stale ⇒ runner died silently.
```

---

## 2. Hard rules — derived from past incidents

1. **Before estimating scope, probe the narrowest capability constraint.** One `curl` call to validate logprob support saves a day of wasted planning.
2. **Read any generator/renderer script before invoking it on new data.** Silent-empty output is the worst failure mode.
3. **Exit 137 is not a script timeout until you've ruled out OOM.** Check `journalctl -k -p err` before blaming `with_timeout`.
4. **HF snapshot dirs use relative symlinks that dangle inside docker mounts. Always `cp -rL` to a real directory first.**
5. **Smoke-test the served endpoint with one `curl` before kicking off a multi-hour eval.**
6. **Pin every lm-eval flag AND task name to the installed version.** Task names are removed/renamed across versions (humaneval and mbpp are gone from 0.4.7). Before invoking `lm-eval --tasks <x>`, run `lm-eval --tasks list | grep <x>` in a throwaway container. `--confirm_run_unsafe_code` exists in ≥0.4.8; `HF_ALLOW_CODE_EVAL=1` is the 0.4.7-compatible path.
7. **Background jobs must stop their servers on failure.** `set -e` without a trap leaks containers on port 8000. Add `trap` + cleanup.
8. **Write `state.json` markers for every completed phase.** Pipeline resumption depends on this.
9. **HF validates `README.md` YAML frontmatter on upload.** `license:` must be on [HF's allowed list](https://huggingface.co/docs/hub/repositories-licenses) — use `mit`, `apache-2.0`, `llama3.1`, `other`, etc. `see-original` is rejected. `render_card.py --license <tag>` is the switch.

---

## 3. Pre-flight checklist (run before every release)

```bash
# Disk
df -h ~ | awk 'NR==2 {print $4, "free"}'                 # Need ≥50G per release
# RAM + GPU
free -h | awk 'NR==2 {print "host:", $7, "avail"}'       # Need ≥60G for BF16 HF-path fallback
nvidia-smi --query-gpu=memory.free --format=csv         # Need model weights + KV budget
# Port
ss -ltn | grep :8000 && echo "PORT 8000 IN USE — resolve"
docker ps --filter publish=8000
# No stale pipeline lock
ls /tmp/spark-lab-*.lock 2>/dev/null
# Images present
docker images | grep -c "tensorrt-llm/release:spark-single-gpu-dev"   # expect ≥1
# HF token if model is gated
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN present" || echo "HF_TOKEN missing"
```

---

## 4. Materialize BF16 weights (mandatory for trtllm-serve)

`trtllm-serve` mounts a single directory as `/workspace/model`. HF cache snapshots (`snapshots/<hash>/`) contain **relative symlinks** into `../../blobs/` that break when only the snapshot dir is mounted. Materialize once per model:

```bash
MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
SNAP=$(ls -d ~/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/*/ | head -1)
DEST=~/spark-lab/models/$(basename $MODEL_ID)-bf16
mkdir -p "$DEST"
cp -rL "$SNAP"/. "$DEST"/
du -sh "$DEST"    # sanity: ~15G for an 8B model
```

---

## 5. Full release flow — step by step

### 5.1. Start the pipeline (NVFP4 quant + serve + eval + perf)

```bash
cd ~/spark-lab
MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
RELEASE_TAG=release-N_$(basename $MODEL_ID)
scripts/pipeline.sh --release "$RELEASE_TAG" --model "$MODEL_ID"
```

This runs: `quantize → snapshot_services → stop_services → eval_nvfp4 → perf → eval_bf16 → render_card → restart_services`.

### 5.2. If `eval_bf16` fails (historical: OOM via HF-transformers path)

Do **not** re-run the phase as-written. Use the API path:

```bash
# a. Ensure BF16 weights materialized (see §4)
DEST=~/spark-lab/models/$(basename $MODEL_ID)-bf16

# b. Serve BF16 via trtllm-serve
NAME=trtllm-bf16-rerun-$$
scripts/serve.sh start "$DEST" "$NAME" 8000
scripts/serve.sh wait 8000

# c. Smoke test
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"quant","prompt":"2+2=","max_tokens":8,"temperature":0}'

# d. Run gsm8k (same task set as NVFP4 for apples-to-apples)
scripts/eval.sh 8000 \
  ~/spark-lab/results/$RELEASE_TAG/eval-bf16 \
  $MODEL_ID v0

# e. Cleanup
scripts/serve.sh stop "$NAME"

# f. Mark phase ok
python3 -c "
import json, pathlib
p = pathlib.Path.home()/'spark-lab/results/$RELEASE_TAG/state.json'
d = json.loads(p.read_text())
d['phases']['eval_bf16'] = 'ok'
p.write_text(json.dumps(d, indent=2))
"
```

Expected wall time: **~3h** for 1319-sample gsm8k at concurrency=1 on GB10.

### 5.3. Render the card

```bash
python3 scripts/render_card.py \
  --release-dir ~/spark-lab/results/$RELEASE_TAG \
  --model-id $MODEL_ID \
  --calibration default
```

Exit codes:
- `0` — clean render, `auto_publish_blocked: false`
- `10` — one or more quality gates triggered; `manifest.json:auto_publish_blocked = true`. **Do not push.** Fix or document the regression.

Current gates (in `render_card.py`):
- `DELTA_FLAG_PCT = 5.0` — >5pp drop on any accuracy task
- `MMLU_FLOOR = 25.0` — below random-chance ⇒ something's broken
- `PPL_CEIL_RATIO = 1.20` — NVFP4 perplexity >1.2× BF16

### 5.4. Post-render verification

```bash
# Read the numbers yourself before shipping
cat ~/spark-lab/results/$RELEASE_TAG/BENCHMARKS.md
jq '.results, .auto_publish_blocked, .quality_gates_triggered' \
   ~/spark-lab/results/$RELEASE_TAG/manifest.json
```

If any section is empty (delta table with zero rows, `results.*` object empty) → extractor bug. Check `render_card.py:extract_results()`.

---

## 6. Task-specific procedures

### 6.1. Generation-only tasks (gsm8k, HumanEval, TriviaQA-generate)

Run via trtllm-serve's OpenAI-compatible `/v1/completions`:

- `eval.sh` covers gsm8k out of the box (task key `v0`).
- HumanEval uses `scripts/run_humaneval.sh` (serves both models in sequence). Requires `HF_ALLOW_CODE_EVAL=1` env var in the eval container. **Do not** pass `--confirm_run_unsafe_code` to lm-eval 0.4.7 — that flag doesn't exist until 0.4.8+.

### 6.2. Logprob tasks (MMLU, HellaSwag, ARC, TruthfulQA, wikitext PPL)

**Currently blocked for NVFP4 on this infrastructure.** `trtllm-serve` returns `400: logprobs is not supported` from its OpenAI shim. HF transformers cannot load NVFP4 weights. vLLM not installed.

To unblock, pick one:
1. Install vLLM with Blackwell support (multi-day effort).
2. Write a custom lm-eval backend against trtllm-serve's native Python API, which does expose logprobs internally.
3. Constrain release cards to generation-only tasks until (1) or (2) is done.

**Until unblocked, release cards are gsm8k + humaneval.** Document this as a known limitation in the card's "Known limitations" section, not as a capability.

### 6.3. Perf sweep

Already scripted as phase in `pipeline.sh` → `perf.sh`. Three concurrency levels (1, 4, 16). Don't bump above 16 without a reason — on GB10, aggregate tok/s plateaus ≥4 and c=16 only inflates p50 latency.

---

## 7. Incident response

### 7.1. "Why did the box reboot?"

Standard triad — run all three, triangulate:

```bash
last -x reboot shutdown | head -10
journalctl --list-boots | tail -5
grep -E "^Start-Date:|^Commandline:" /var/log/apt/history.log | tail -20
```

Look for: kernel version jumps across boots, `apt upgrade` + kernel package installs matching reboot timestamp, OOM/panic signatures in the previous boot's journal (`journalctl -b -1 -p err`).

### 7.2. "Pipeline phase died with exit 137"

This looks like a `with_timeout` hit, but often isn't:

```bash
PHASE_FAIL_TIME="2026-04-23 04:23"   # from events.log
journalctl -k --since "$PHASE_FAIL_TIME -0:30:00" --until "$PHASE_FAIL_TIME +0:05:00" \
  | grep -iE "oom|killed process|memory pressure"
```

If OOM killer fired anywhere in that window, the cause is RAM pressure, not the 6h timer. Root-cause the memory hog (often another user service competing for host RAM during the eval).

### 7.3. "trtllm-serve won't start"

Check container log:

```bash
docker logs $(docker ps -aqf "name=trtllm") 2>&1 | tail -40
```

Common failures:
- `Unrecognized model in /workspace/model. Should have a model_type key in config.json` → **snapshot-symlink problem.** Materialize with `cp -rL` (§4).
- `ValueError: logprobs is not supported` → you tried to use logprobs against the OpenAI shim. See §6.2.
- CUDA out of memory → reduce `--max_batch_size` in `serve.sh` or stop the competing GPU process.

### 7.4. "Background script died and leaked a docker container"

`set -e` without a trap leaks. Always run:

```bash
docker ps --filter name=trtllm
# If orphaned:
docker stop <name>
```

Add to any new background script:

```bash
cleanup() { scripts/serve.sh stop "$NAME" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
```

---

## 8. Release readiness checklist

Before `git push` / `huggingface-cli upload`:

- [ ] `manifest.json:auto_publish_blocked == false`
- [ ] `BENCHMARKS.md` delta table has ≥1 row with real numbers (not empty)
- [ ] `README.md` has no `{args.MODEL_ID}` literal leakage
- [ ] `REPRODUCE.sh` is executable (`chmod 755`) and references the pinned `modelopt_branch`
- [ ] `state.json` shows **every** expected phase as `ok`
- [ ] Quant weight files hashed in `manifest.json:weights_sha256` (count matches `ls quant/saved_models_*/ | wc -l`)
- [ ] Any asymmetric task coverage (e.g., gsm8k-only) documented in README's "Known limitations"
- [ ] Session-retro run to capture any new lessons → update this SOP

---

## 9. Useful one-liners

```bash
# Show release state at a glance
cat ~/spark-lab/results/$RELEASE_TAG/state.json | jq

# Tail any active eval
ls -t ~/spark-lab/logs/eval-*.log | head -1 | xargs tail -f

# Count progress samples in a running eval
grep -c "Requesting API:" $(ls -t ~/spark-lab/logs/eval-*.log | head -1)

# Watch all trtllm containers
watch -n 5 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep trtllm'

# Force-stop every trtllm container (nuclear)
docker ps --filter name=trtllm -q | xargs -r docker stop
```

---

## 10. Known infrastructure constraints (as of 2026-04-23)

| Constraint | Impact | Workaround |
|---|---|---|
| `trtllm-serve` OpenAI shim does not expose logprobs | Cannot run MMLU/HellaSwag/ARC/wikitext on NVFP4 | Generation-only tasks, or custom trtllm Python adapter |
| No vLLM installed | Same as above | Install with Blackwell support (non-trivial on GB10) |
| HF cache uses relative symlinks | Snapshots don't mount cleanly into docker | `cp -rL` (§4) |
| Host RAM ~119GB shared with services | HF-transformers path can OOM with MMLU+wikitext | Use API path (§5.2) |
| Other long-running user services competing for host RAM | Can trigger OOM mid-eval | Either pause unrelated services during multi-hour evals, or use API path to keep weights on GPU |
| **BBH + R1-Distill + trtllm-serve + aiohttp**: hits `asyncio.TimeoutError` mid-run even with `--gen_kwargs max_gen_toks=1024`. Long CoT generations exceed aiohttp's internal per-request timeout. | Cannot run BBH via lm-eval's `local-completions` on R1-Distill quants. Observed twice 2026-04-23. | Untested: `num_concurrent=1`, explicit `timeout=` model_arg. Low-effort alternative: use gsm8k-only until a logprob-capable backend is available. |

---

## 11. Changelog

### v0.2 — 2026-04-23 (release-0.1)
- Per-task quality gate thresholds (math/reasoning tightened to 3pp; see `TASK_DELTA_THRESHOLDS_PCT` in `render_card.py`).
- Minimum-coverage gate: don't auto-publish if zero benchmark families completed.
- 95% CI for the BF16↔NVFP4 delta surfaced in BENCHMARKS.md (unpaired normal approximation; paired bootstrap deferred to release-1 pending `--log_samples` re-run).
- Full provenance in `manifest.json`: container digest, upstream base-model SHA, `spark-nvfp4-lab` git SHA, self-sha256 of `render_card.py`, eval context length, calibration metadata.
- Card enrichments: headline delta on the card itself, on-disk + observed peak VRAM comparison, known-good `trtllm-serve` command, recommended-use block, tested/untested engine list, verify-this-release recipe, explicit "tasks NOT evaluated" section.
- HF YAML override: explicit `tags: [nvfp4, fp4, 4-bit, tensorrt-llm, modelopt, …]` to bypass HF's auto `8-bit` misclassification on ModelOpt packed weights.
- `--override-gate-exit` flag on `render_card.py` for documented manual overrides (records reason in manifest).
- CI: GitHub Actions runs shellcheck (warning-level) on all `scripts/*.sh` and `tests/*.sh`, plus a fixture-based functional test of `render_card.py` that asserts invariants on the rendered README/BENCHMARKS/manifest.
- SOP rule #9 added: HF `README.md` YAML license must be on HF's allowed list; `render_card.py --license <tag>` is the switch.

### v0.1 — 2026-04-23 (release-0)
- Initial SOP derived from release-0 session (DeepSeek-R1-Distill-Llama-8B).
- 8 hard rules, incident response procedures, 10 known constraints.
- Companion to first public artifact: [`GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4`](https://huggingface.co/GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4).

---

*Update after every release retro. Increment the version in the frontmatter of
this file and add a dated entry above.*
