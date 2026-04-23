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
10. **HF YAML `tags:` are additive, not replacement.** Listing `nvfp4, fp4, 4-bit` does **not** remove HF's auto-applied `8-bit` tag (which fires on ModelOpt packed weights because they contain FP8 scales). Explicit tags still help discovery; plan on the auto-tag sticking until HF fixes the detector or we restructure the scale layout.
11. **Provenance capture is lossy after the fact.** Container digest, ModelOpt commit, calibration sample hash, and upstream model revision must be recorded **at quant time** (in `quantize.sh`), not retrofitted at render time. Anything not captured during the run is best-effort only — honestly mark it `"not-captured-release-N"` in the manifest.
12. **Unpaired 95% CI on the BF16↔NVFP4 delta is a conservative upper bound.** Both evals hit the same gsm8k samples, so a paired bootstrap would be tighter. Re-run with `lm-eval --log_samples` to enable paired analysis. Until then, quote the unpaired CI and note the method.
13. **Second-pass verification is mandatory before any public push.** After you think you're done, re-read every changed file and run the assertion script. False confidence on first pass is cheap to correct; a broken public card is not. The session of 2026-04-23 caught a perf-interpretation bug (claimed saturation at c=16 when actual was c=4) only on the second pass.
14. **Exit code 0 is not "it worked."** A `docker run ... | tee | tail` pipeline returns the exit code of `tail`, not the docker. A bash script with `set -e` returns 0 if the last line is `echo done`. Always verify the actual artifact (results.json written, HF API shows new commit, page returns 200) — not just the exit code.

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
  --calibration default \
  --license mit \
  --eval-context-length 2048 \
  --bf16-weights-dir ~/spark-lab/models/<model>-bf16 \
  --bf16-vram-observed-gib <N> \
  --nvfp4-vram-observed-gib <N> \
  --tested-engines "TensorRT-LLM 1.1.0rc3 (trtllm-serve)" \
  --untested-engines "vLLM (Blackwell), SGLang"
```

Exit codes:
- `0` — clean render, `auto_publish_blocked: false`
- `10` — one or more quality gates triggered; `manifest.json:auto_publish_blocked = true`. **Do not push** unless you add `--override-gate-exit "<reason>"`, which records the reason in `manifest.json:gate_override_reason` and exits 0. Use sparingly and only with a clear written justification.

Current gates (in `render_card.py`):
- **Per-task accuracy thresholds** via `TASK_DELTA_THRESHOLDS_PCT`:
  - Math / reasoning / code (`gsm8k`, `mmlu`, `math*`, `humaneval`, `mbpp`, `bbh`): `|Δ| ≤ 3pp`
  - Commonsense / knowledge (`hellaswag`, `arc*`, `truthfulqa`, `winogrande`): `|Δ| ≤ 5pp`
  - Anything else: `DEFAULT_DELTA_THRESHOLD_PCT = 5pp`
- `MMLU_FLOOR_PCT = 25.0` — below random-chance ⇒ something's broken
- `PPL_CEIL_RATIO = 1.20` — NVFP4 perplexity > 1.2× BF16
- `MIN_BENCHMARK_FAMILIES = 1` — do not auto-publish if zero tasks completed

**Measuring VRAM for the card.** Spin up the trtllm-serve container, hit it with one completion request (to force KV cache allocation), and read `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`. Do both BF16 and NVFP4. This is a ~2-minute step; skipping it means the VRAM lines on the card are honest-unknown rather than honest-observed.

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

Before `git push` / `hf upload`:

**Artifact sanity**
- [ ] `manifest.json:auto_publish_blocked == false`, OR `--override-gate-exit "<reason>"` was passed AND the reason is recorded in `manifest.json:gate_override_reason`.
- [ ] `BENCHMARKS.md` delta table has ≥1 row with real numbers (not empty).
- [ ] `README.md` has no `{args.*}` literal template leakage.
- [ ] `README.md` YAML `license:` is on HF's allowed list (mit, apache-2.0, llama3.1, other, …).
- [ ] `REPRODUCE.sh` is executable (`chmod 755`) and references the pinned `modelopt_branch` + `container_digest`.
- [ ] `state.json` shows **every** expected phase as `ok`.
- [ ] Weight files hashed in `manifest.json:weights_sha256` (count matches `ls quant/saved_models_*/ | wc -l`).

**Provenance completeness (manifest.json)**
- [ ] `base_model.sha` set (fetched from HF API).
- [ ] `tooling.container_digest` set (`docker image inspect`).
- [ ] `tooling.trtllm_version` recorded.
- [ ] `tooling.spark_nvfp4_lab_git_sha` recorded (should match current `HEAD`).
- [ ] `tooling.render_card_py_sha256` recorded (self-hash).
- [ ] `calibration.dataset`, `calibration.n_samples_default`, `calibration.seq_len_default` present.
- [ ] `eval.context_length`, `eval.lm_eval_version`, `eval.bootstrap_iters`, `eval.ci_method` present.
- [ ] `footprint.bf16_bytes`, `footprint.nvfp4_bytes`, `footprint.compression_ratio` populated.
- [ ] Observed VRAM values passed via CLI if measured (else left `null` with a note in the card).

**Card surface completeness (README.md)**
- [ ] **Headline** line at the top has the actual delta number, not a "see below".
- [ ] Upstream base-model link appears within the first 10 lines.
- [ ] "Known-good usage" section has a copy-paste `docker run` command.
- [ ] "Tasks not evaluated" section explains what was skipped and why (not just omits it).
- [ ] Context-length caveat present if eval ran at <16k.
- [ ] Tested vs untested engines listed explicitly.
- [ ] "How to verify this release" block present.

**Process**
- [ ] Session-retro run to capture any new lessons → update this SOP (§11 Changelog).
- [ ] Second-pass verification script run (`bash tests/test_render_card.sh` must pass against the fixture; then re-read the live files with eyes).
- [ ] CI green on `main` before the HF push (`gh api repos/<owner>/spark-nvfp4-lab/actions/runs --jq '.workflow_runs[0] | {status,conclusion}'`).

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
| **HF YAML `tags:` are additive, not replacement.** HF's auto-tagger adds `8-bit` to NVFP4 ModelOpt packed weights (because FP8 scales are detected), even when YAML explicitly lists `nvfp4/fp4/4-bit`. Observed on `GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4` post-release-0.1. | Wrong bit-width tag hurts discovery of the release under `4-bit` search facets. | Document the quirk in the release card; file HF issue if persistent; long-term fix requires either HF auto-tagger update or restructured scale storage. |
| **humaneval + mbpp not in `lm-eval-harness 0.4.7`.** Tasks were removed/renamed upstream. Installing `lm-eval[api]==0.4.7` does not register them. | Cannot add code-benchmarks to the card without extra setup. | `bigcode-evaluation-harness` (separate tool, unknown trtllm-serve compat) OR upgrade to `lm-eval>=0.4.8`. |
| **`lm-eval local-completions` does not expose `timeout=` model-arg in 0.4.7.** Per-request aiohttp timeout hardcoded to ~5 min. | Long-CoT reasoning models (R1, o1-like) hit this on BBH/MATH. | Wait for upstream fix, patch locally, or cap generation via `--gen_kwargs max_gen_toks=<N>` — note that this alone did **not** resolve the BBH timeout in the 2026-04-23 session. |
| **HTTPS git push to GitHub may return HTTP 500 intermittently on large first pushes.** Hit once on 2026-04-23 release-0.1 commit (+688/-89). | Push appears to fail but server-side may or may not have rejected. | Retry with `git config http.version HTTP/1.1 && git config http.postBuffer 524288000`. Verify with `gh api repos/<owner>/<repo>/commits/main --jq .sha` before reattempting. |

---

## 11. Changelog

### v0.2 — 2026-04-23 (release-0.1)

**Renderer changes:**
- Per-task quality gate thresholds (math/reasoning tightened to 3pp; see `TASK_DELTA_THRESHOLDS_PCT` in `render_card.py`).
- Minimum-coverage gate: don't auto-publish if zero benchmark families completed.
- 95% CI for the BF16↔NVFP4 delta surfaced in BENCHMARKS.md (unpaired normal approximation; paired bootstrap deferred pending `--log_samples` re-run).
- Full provenance in `manifest.json`: container digest, upstream base-model SHA, `spark-nvfp4-lab` git SHA, self-sha256 of `render_card.py`, eval context length, calibration metadata.
- Card enrichments: headline delta on the card itself, on-disk + observed peak VRAM comparison, known-good `trtllm-serve` command, recommended-use block, tested/untested engine list, verify-this-release recipe, explicit "tasks NOT evaluated" section, chat template reference.
- HF YAML tags expanded: explicit `nvfp4, fp4, 4-bit, tensorrt-llm, modelopt, blackwell, gb10`.
- `--override-gate-exit "<reason>"` flag for documented manual overrides (records reason in manifest.json).
- Perf-table interpretation picks the true saturation concurrency (smallest `c` within 5% of peak aggregate), not max concurrency. Previously reported c=16 as saturation when the real answer was c=4.

**Repo infrastructure:**
- `LICENSE` file (MIT).
- `.github/workflows/ci.yml`: ShellCheck (warning level) on `scripts/*.sh` + `tests/*.sh`; fixture-based functional test of `render_card.py` asserting 27 invariants on rendered output.
- `tests/fixtures/release-sample/` synthetic release dir + `tests/test_render_card.sh` assertion harness.
- `README.md` CI badge, two quickstart paths (quantize-your-own, serve-an-existing-release).
- SOP §11 changelog format instituted.

**Hard rules added (§2):**
- #10: HF YAML tags are additive, not replacement.
- #11: Provenance capture is lossy after the fact.
- #12: Unpaired 95% CI is a conservative upper bound; paired requires `--log_samples`.
- #13: Second-pass verification is mandatory before public push.
- #14: Exit 0 ≠ "it worked" — verify the artifact independently.

**Process changes:**
- §8 readiness checklist expanded to 3 sub-sections (artifact sanity / provenance / card surface completeness).
- §5.3 render command example updated with all new CLI flags.
- §10 constraints table +4 rows (HF auto-tagger quirk, lm-eval 0.4.7 missing humaneval/mbpp, `local-completions` missing `timeout=`, GitHub HTTP 500 on large first pushes).

**Release-0.1 artifact:** [`GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4`](https://huggingface.co/GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4) commit `2026-04-23T16:05:40Z` (weights unchanged from release-0; only card + manifest refreshed).

### v0.1 — 2026-04-23 (release-0)
- Initial SOP derived from release-0 session (DeepSeek-R1-Distill-Llama-8B).
- 8 hard rules, incident response procedures, 10 known constraints.
- Companion to first public artifact: [`GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4`](https://huggingface.co/GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4).

---

## 12. Release planning — NOW vs LATER triage

Every batch of improvements (post-release reviews, new reviewer feedback, etc.) gets split into two buckets before any work starts:

- **NOW** = code, docs, HF card, repo, CI. Does not require re-running quant or eval. Deploys via file edits + `git push` + `hf upload`.
- **LATER** = requires a fresh quant and/or eval pass (minimum ~4h wall on a DGX Spark). Bundled into the next tuning-run session.

Before agreeing to any scope expansion, ask: *does this require re-running a multi-hour pipeline?* If yes, it's LATER. If no, it's a NOW candidate, but still pre-flighted (§3).

### 12.1. Release-1 targets (deferred from release-0.1)

Carried forward from the 2026-04-23 reviewer pass:

**Re-eval required:**
- [ ] **`--log_samples` on both eval passes**, then compute paired bootstrap CI on the delta. Tighter than the unpaired normal approximation shipped in release-0.1.
- [ ] **MATH-500** on both BF16 and NVFP4 (generation-only, works through existing API path).
- [ ] **AIME subset** if time permits (generation-only).
- [ ] **Re-run at context length ≥ 16k** to test whether the gsm8k regression narrows for R1-Distill.
- [ ] **Matched-workload perf sweep** (same `n_requests` at each concurrency level).
- [ ] **Side-by-side sample generations** (10–20 prompts, BF16 vs NVFP4 output in BENCHMARKS.md).

**Re-quant required:**
- [ ] **Calibration sample hash captured at quant time** in `quantize.sh` → `quant/manifest.json`. Removes the "not-captured-release-0" placeholder.
- [ ] **ModelOpt commit SHA** captured by `quantize.sh` (currently only branch pin).
- [ ] **`max_seq_len` used during quant** recorded.
- [ ] *(Experiment)* Reasoning-appropriate calibration corpus (GSM8K or MBPP samples vs default cnn_dailymail) — one-off test on DeepSeek-R1-Distill-Llama-8B to see if CoT preservation improves.

**Infrastructure work (~1 day each):**
- [ ] **Custom `trtllm` Python backend for lm-eval** exposing logprobs. Unlocks MMLU / HellaSwag / ARC / TruthfulQA / wikitext-PPL across every future release. Highest-leverage item on this list.
- [ ] **Alternative: install vLLM with Blackwell support** on the lab box.
- [ ] *(Aspirational)* NVFP4 + quantized KV cache exploration for long-context perf.

**Pipeline generality:**
- [ ] **Test on a second model** (denser 8B or 14B) to validate the pipeline on something other than R1-Distill.

### 12.2. What we explicitly chose NOT to pursue

- **HumanEval via lm-eval-harness** — task removed/renamed in 0.4.7. Either upgrade the harness (may break other things) or adopt `bigcode-evaluation-harness` (separate tool, unknown trtllm-serve compat). Neither is a small task.
- **BBH via `local-completions`** — burns on aiohttp timeout at ~45% progress despite `max_gen_toks` cap. Needs `num_concurrent=1` testing and/or upstream lm-eval change. Low ROI vs MATH-500.
- **Full-MMLU via HF transformers backend** — works in principle but OOM-risky and not apples-to-apples against NVFP4 (which can't run HF transformers). Supplanted by the custom trtllm Python backend item above.
- **Multi-seed runs on gsm8k** — eval is already deterministic at temperature=0; different seeds just reshuffle few-shot examples. Low signal. Paired bootstrap CI is the higher-leverage statistical upgrade.
- **Framing the accuracy delta positively** ("strong reasoning retention: −3.94 pt") — breaks the "honest numbers" positioning. The card states the number.
- **"Contributing" scaffolding** — premature for a one-person lab with zero external interest signals.

---

## 13. Methodology reference

### 13.1. Bootstrap CI on the quantization delta

lm-eval-harness writes per-task `exact_match_stderr` (or equivalent) computed via internal bootstrap at `bootstrap_iters=100000`. The stderr is the standard error of the mean accuracy estimate for that single run.

**Unpaired (what we ship in release-0.1):**
```
SE(Δ) = sqrt(SE_bf16² + SE_nvfp4²)
95% CI ≈ Δ ± 1.96 · SE(Δ)
```
This assumes the two runs are independent. Because both hit the same samples, it's a conservative **upper bound** on uncertainty.

**Paired (release-1 target):**
Requires `lm-eval --log_samples` to capture per-sample correctness vectors for both runs. Compute per-sample difference vector `d_i = correct_nvfp4[i] - correct_bf16[i]`, bootstrap-resample to estimate `SE(d̄)`, which will be tighter because it removes between-sample variance.

### 13.2. Quality-gate rationale

Different task families tolerate quantization differently:

- **Math / reasoning / code** (gsm8k, MATH, humaneval, BBH, mbpp): low noise floor, sensitive to numerical precision loss. A 3pp drop is meaningful. Gate at 3pp.
- **Commonsense / MCQ** (HellaSwag, ARC, TruthfulQA, Winogrande): noisier ground-truth, lower ceiling gap between model generations. 5pp is appropriate.
- **MMLU**: special-cased floor at 25% (random-chance for 4-option MCQ) — below this means the model fundamentally broke.
- **Perplexity** (wikitext): ratio-based gate at 1.2× — small absolute PPL increases can mean large distribution shifts.

These are defaults in `TASK_DELTA_THRESHOLDS_PCT`; override per-release when justified.

### 13.3. Perf-table interpretation

For a throughput sweep across concurrency levels, "saturation" is the smallest concurrency within 5% of peak aggregate throughput. Higher concurrencies past the saturation point trade single-stream latency for no meaningful aggregate gain — worse user-perceived experience. Report both and name the saturation concurrency explicitly in the card.

---

*Update after every release retro. Increment the version in the frontmatter of
this file and add a dated entry above.*
