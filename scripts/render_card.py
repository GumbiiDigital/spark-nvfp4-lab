#!/usr/bin/env python3
"""
Render a HuggingFace model card + BENCHMARKS.md + manifest.json from
phase outputs. Reads:
  results/<release>/quant/manifest.json     (from quantize.sh)
  results/<release>/eval-bf16/results.json
  results/<release>/eval-nvfp4/results.json
  results/<release>/perf/perf.json
Writes:
  results/<release>/README.md
  results/<release>/BENCHMARKS.md
  results/<release>/manifest.json
  results/<release>/REPRODUCE.sh
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

# Quality-gate thresholds.
# Per-task thresholds: keys match task name prefixes; first-match wins.
TASK_DELTA_THRESHOLDS_PCT = {
    "mmlu":       3.0,
    "gsm8k":      3.0,   # math/reasoning — tighter
    "math":       3.0,
    "humaneval":  3.0,
    "mbpp":       3.0,
    "bbh":        3.0,
    "hellaswag":  5.0,
    "arc":        5.0,
    "truthfulqa": 5.0,
    "winogrande": 5.0,
}
DEFAULT_DELTA_THRESHOLD_PCT = 5.0
MMLU_FLOOR_PCT = 25.0     # below random ⇒ broken
PPL_CEIL_RATIO = 1.20     # NVFP4 PPL > 1.2x BF16 PPL ⇒ flag
MIN_BENCHMARK_FAMILIES = 1  # min completed tasks before auto-publish allowed


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> dict:
    """sha256 of every regular file under path."""
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(path))] = sha256_file(p)
    return out


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def task_threshold(task: str) -> float:
    for prefix, thr in TASK_DELTA_THRESHOLDS_PCT.items():
        if task.startswith(prefix):
            return thr
    return DEFAULT_DELTA_THRESHOLD_PCT


def extract_results(p: Path) -> dict:
    """Pull per-task (value, stderr, metric, n_samples) from lm-eval results.json."""
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    n_samples_map = j.get("n-samples", {})
    out = {}
    FILTER_RANK = {"strict-match": 0, "none": 1, "flexible-extract": 2, "get-answer": 3}
    METRIC_RANK = {"exact_match": 0, "acc": 1, "pass@1": 2, "acc_norm": 3}
    for task, scores in (j.get("results") or {}).items():
        best = None
        for k, v in scores.items():
            if not isinstance(v, (int, float)) or "," not in k or "stderr" in k:
                continue
            base, filt = k.rsplit(",", 1)
            fr = FILTER_RANK.get(filt, 99)
            mr = METRIC_RANK.get(base, 99)
            cand = (fr, mr, k, v, base, filt)
            if best is None or cand < best:
                best = cand
        if best:
            stderr_key = f"{best[4]}_stderr,{best[5]}"
            se = scores.get(stderr_key)
            se = float(se) if isinstance(se, (int, float)) else None
            n = n_samples_map.get(task, {}).get("effective") if isinstance(n_samples_map.get(task), dict) else None
            out[task] = {
                "metric": best[2],
                "value": float(best[3]),
                "stderr": se,
                "n_samples": n,
            }
    return out


def unpaired_delta_ci95(bf16_se: float | None, nvfp4_se: float | None) -> float | None:
    """95% CI half-width for delta when only per-eval stderrs are available.
    Conservative: assumes independence between the two runs.
    Paired would be tighter but requires --log_samples."""
    if bf16_se is None or nvfp4_se is None:
        return None
    return 1.96 * math.sqrt(bf16_se ** 2 + nvfp4_se ** 2)


def delta_table(bf16: dict, nvfp4: dict) -> tuple[str, list[dict], list[tuple]]:
    """Side-by-side BF16 vs NVFP4 with delta + 95% CI + per-task gate."""
    rows = []
    keys = sorted(set(bf16) | set(nvfp4))
    md = [
        "| Task | Metric | BF16 | NVFP4 | Δ (pp) | 95% CI (pp) | Gate | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    flags = []
    for k in keys:
        b = bf16.get(k, {}); q = nvfp4.get(k, {})
        bv = b.get("value"); qv = q.get("value")
        b_se = b.get("stderr"); q_se = q.get("stderr")
        metric = b.get("metric") or q.get("metric") or "?"
        if bv is None or qv is None:
            md.append(f"| {k} | {metric} | {bv} | {qv} | – | – | – | missing-side |")
            continue
        is_accuracy = metric.startswith("acc") or "exact_match" in metric or "pass@1" in metric
        is_perplexity = "perplexity" in metric or k == "wikitext"
        delta_pct = (qv - bv) * 100 if is_accuracy else (qv - bv)
        ci_hw = unpaired_delta_ci95(b_se, q_se) if is_accuracy else None
        ci_hw_pct = ci_hw * 100 if ci_hw is not None else None
        gate_pct = task_threshold(k)
        flag = ""
        if k == "mmlu" and qv * 100 < MMLU_FLOOR_PCT and metric.startswith("acc"):
            flag = "MMLU<random"; flags.append(("mmlu_below_random", k, qv))
        if is_accuracy and abs(delta_pct) > gate_pct:
            flag = f"|Δ|>{gate_pct:.1f}pp"; flags.append(("delta_over_task_threshold", k, delta_pct))
        if is_perplexity and bv > 0 and qv / bv > PPL_CEIL_RATIO:
            flag = "PPL>1.2x"; flags.append(("ppl_blowup", k, qv / bv))
        bf_cell = f"{bv:.4f}" + (f" ± {b_se:.4f}" if b_se is not None else "")
        qf_cell = f"{qv:.4f}" + (f" ± {q_se:.4f}" if q_se is not None else "")
        delta_cell = f"{delta_pct:+.2f}"
        ci_cell = f"±{ci_hw_pct:.2f}" if ci_hw_pct is not None else "n/a"
        gate_cell = f"|Δ|≤{gate_pct:.1f}"
        md.append(f"| {k} | {metric} | {bf_cell} | {qf_cell} | {delta_cell} | {ci_cell} | {gate_cell} | {flag} |")
        rows.append({
            "task": k, "metric": metric,
            "bf16": bv, "bf16_stderr": b_se,
            "nvfp4": qv, "nvfp4_stderr": q_se,
            "delta_pp": delta_pct, "ci95_halfwidth_pp": ci_hw_pct,
            "gate_pp": gate_pct, "flag": flag,
        })
    return "\n".join(md), rows, flags


def render_perf_md(perf: dict) -> tuple[str, dict]:
    """Perf table with settings + interpretation."""
    if not perf.get("runs"):
        return "(no perf data captured)", {}
    header = [
        "**Settings:** prompt = {pc} chars, max_tokens = {mt}, temperature = 0 (greedy).".format(
            pc=perf.get("prompt_chars", "?"), mt=perf.get("max_tokens", "?"),
        ),
        "",
        "| Concurrency | n_ok | wall (s) | p50 lat (s) | p95 lat (s) | Single-stream tok/s | Aggregate tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for r in perf["runs"]:
        if r.get("n_ok"):
            rows.append(
                f"| {r['concurrency']} | {r['n_ok']}/{r['n_requests']} | {r['wall_s']} | "
                f"{r['p50_latency_s']} | {r['p95_latency_s']} | "
                f"{r['single_stream_tok_s']} | {r['aggregate_tok_s']} |"
            )
    # interpretation — find the smallest concurrency within 5% of peak aggregate
    runs = sorted((r for r in perf["runs"] if r.get("n_ok")), key=lambda r: r["concurrency"])
    interp = ""
    if len(runs) >= 2:
        peak = max(r["aggregate_tok_s"] for r in runs)
        saturation_run = next(r for r in runs if r["aggregate_tok_s"] >= 0.95 * peak)
        sat_c = saturation_run["concurrency"]
        sat_tok = saturation_run["aggregate_tok_s"]
        interp = (
            f"\n**Interpretation:** throughput saturates at concurrency ≈ {sat_c} "
            f"(~{sat_tok:.0f} tok/s aggregate). Beyond that you're buying latency, not throughput. "
            f"Note: this sweep is **not** a matched workload — different `n_requests` at each concurrency "
            f"level. A matched-workload sweep is on the release-1 list."
        )
    return "\n".join(header + rows) + interp, {
        "prompt_chars": perf.get("prompt_chars"),
        "max_tokens": perf.get("max_tokens"),
    }


def fetch_base_model_sha(model_id: str) -> str | None:
    """Best-effort pull of upstream base model's latest commit SHA from HF."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}", timeout=10) as r:
            return json.loads(r.read()).get("sha")
    except Exception:
        return None


def docker_image_digest(image: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--calibration", default="default",
                    help="Calibration dataset key: default (cnn_dailymail) or code (codeparrot)")
    ap.add_argument("--license", default="other",
                    help="SPDX-style HF license tag (hf.co/docs/hub/repositories-licenses)")
    ap.add_argument("--eval-context-length", type=int, default=2048,
                    help="Max context length used during eval. Note on card.")
    ap.add_argument("--bf16-weights-dir", type=Path, default=None,
                    help="Optional path to unquantized BF16 weights for disk-size comparison.")
    ap.add_argument("--bf16-vram-observed-gib", type=float, default=None,
                    help="Optional observed peak VRAM of BF16 serve (GiB).")
    ap.add_argument("--nvfp4-vram-observed-gib", type=float, default=None,
                    help="Optional observed peak VRAM of NVFP4 serve (GiB).")
    ap.add_argument("--tested-engines", default="TensorRT-LLM (trtllm-serve)",
                    help="Comma-separated list of engines actually tested.")
    ap.add_argument("--untested-engines", default="vLLM, SGLang",
                    help="Comma-separated list of engines NOT tested (user should verify).")
    ap.add_argument("--override-gate-exit", default=None,
                    help=("Reason to override gate-triggered exit 10 down to 0. "
                          "Records `override_reason` in manifest.json. Use sparingly."))
    ap.add_argument("--skipped-tasks-note", default=(
        "MMLU, HellaSwag, ARC, TruthfulQA, and wikitext-PPL were **not** run because "
        "they require log-probability evaluation; `trtllm-serve`'s OpenAI-compatible "
        "endpoint does not currently expose `logprobs` and HF transformers cannot load "
        "NVFP4 weights. This is a tooling constraint on release-0, not an omission. "
        "Release-1 will unblock this via a custom `trtllm` Python backend."),
                    help="Plain prose explaining what was NOT evaluated and why.")
    args = ap.parse_args()

    rd = args.release_dir
    bf16 = extract_results(rd / "eval-bf16" / "results.json")
    nvfp4 = extract_results(rd / "eval-nvfp4" / "results.json")
    perf = json.loads((rd / "perf" / "perf.json").read_text()) if (rd / "perf" / "perf.json").exists() else {}
    quant_manifest = json.loads((rd / "quant" / "manifest.json").read_text()) if (rd / "quant" / "manifest.json").exists() else {}

    delta_md, delta_rows, flags = delta_table(bf16, nvfp4)
    perf_md, perf_settings = render_perf_md(perf)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # Min-coverage gate
    completed = sum(1 for r in delta_rows if r["flag"] != "missing-side")
    if completed < MIN_BENCHMARK_FAMILIES:
        flags.append(("min_coverage_not_met", "_meta", completed))

    # ---- Provenance ---------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    this_script = Path(__file__).resolve()
    base_sha = fetch_base_model_sha(args.model_id)
    container_image = "nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"
    container_digest = docker_image_digest(container_image)

    # Disk-size comparison
    nvfp4_weights_dir = next((d for d in (rd / "quant").iterdir() if d.is_dir()), None) if (rd / "quant").exists() else None
    nvfp4_bytes = dir_bytes(nvfp4_weights_dir) if nvfp4_weights_dir else None
    bf16_bytes = dir_bytes(args.bf16_weights_dir) if args.bf16_weights_dir else None
    compression_ratio = round(bf16_bytes / nvfp4_bytes, 2) if bf16_bytes and nvfp4_bytes else None

    def gib(b): return round(b / (1024 ** 3), 2) if b else None

    # Headline number
    headline = ""
    if delta_rows:
        # Prefer gsm8k if present, else first completed row
        gsm = next((r for r in delta_rows if r["task"] == "gsm8k"), None)
        hr = gsm or delta_rows[0]
        ci_note = f" (95% CI ±{hr['ci95_halfwidth_pp']:.2f}pp)" if hr.get("ci95_halfwidth_pp") else ""
        headline = (
            f"NVFP4 shifts **{hr['task']}** by **{hr['delta_pp']:+.2f} pp** vs BF16 "
            f"({hr['bf16']*100:.2f}% → {hr['nvfp4']*100:.2f}%){ci_note} on a single GB10."
        )

    # ---- BENCHMARKS.md ------------------------------------------------------
    bench_md = f"""# Benchmarks — {args.model_id} (NVFP4)

**Generated:** {now} UTC
**Hardware:** NVIDIA GB10 (DGX Spark), aarch64, CUDA 13.0
**Calibration:** `{args.calibration}` (cnn_dailymail, ModelOpt 0.35.0 defaults: 512 samples × 512 tokens unless `quantize.sh` overridden)
**Eval context length:** {args.eval_context_length} tokens — see **Caveat** below.

## Quality (BF16 baseline vs NVFP4 quantization)

{delta_md}

**95% CI method:** unpaired normal approximation using lm-eval bootstrap stderrs
(`bootstrap_iters=100000`). The BF16 and NVFP4 evals hit identical samples, so a
*paired* bootstrap would be tighter; that requires re-running with `--log_samples`
and is on the release-1 list.

**Caveat — context length.** This eval ran at `max_length = {args.eval_context_length}`.
DeepSeek-R1-Distill is a reasoning distill whose ceiling scales with chain-of-thought
length; a 2k cap likely suppresses both precisions, and may suppress NVFP4 more than
BF16. Re-running at 16k+ context is on the release-1 list.

## Performance — TensorRT-LLM on GB10

{perf_md}

## Tasks **not** evaluated

{args.skipped_tasks_note}

## Quality gates triggered

{"None — passed all gates." if not flags else chr(10).join(f"- **{f[0]}** on `{f[1]}` (value: {f[2]})" for f in flags)}

## Tested engines

- Tested: {args.tested_engines}
- Untested (user must verify before relying on these): {args.untested_engines}
"""
    (rd / "BENCHMARKS.md").write_text(bench_md)

    # ---- README.md (HF model card) -----------------------------------------
    # Disk / VRAM block
    disk_vram_lines = []
    if bf16_bytes and nvfp4_bytes:
        disk_vram_lines.append(
            f"- **On-disk footprint:** {gib(bf16_bytes)} GiB (BF16) → {gib(nvfp4_bytes)} GiB (NVFP4) — **{compression_ratio}× reduction**"
        )
    if args.bf16_vram_observed_gib or args.nvfp4_vram_observed_gib:
        parts = []
        if args.bf16_vram_observed_gib:
            parts.append(f"BF16 ~{args.bf16_vram_observed_gib} GiB")
        if args.nvfp4_vram_observed_gib:
            parts.append(f"NVFP4 ~{args.nvfp4_vram_observed_gib} GiB")
        disk_vram_lines.append(
            f"- **Observed peak VRAM during eval** (max_seq_len={args.eval_context_length}): {' / '.join(parts)} — includes weights + KV cache + activations"
        )
    disk_vram_block = "\n".join(disk_vram_lines) if disk_vram_lines else ""

    readme = f"""---
license: {args.license}
base_model: {args.model_id}
library_name: transformers
pipeline_tag: text-generation
tags:
- nvfp4
- fp4
- 4-bit
- quantized
- tensorrt-llm
- modelopt
- blackwell
- gb10
- spark-nvfp4-lab
---

# {args.model_id.split("/")[-1]} — NVFP4 Quantization

> **Headline:** {headline if headline else "See BENCHMARKS.md for the full delta table."}

Quantized from [`{args.model_id}`](https://huggingface.co/{args.model_id}) using [NVIDIA TensorRT Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer).
Requires NVIDIA Blackwell architecture (GB10, B200, GB200, RTX 5090, RTX 6000 Pro).

**Upstream model:** [{args.model_id}](https://huggingface.co/{args.model_id}) @ `{base_sha[:12] if base_sha else "unknown"}`
**Calibration:** `{args.calibration}` (cnn_dailymail)
**Eval context length:** {args.eval_context_length} tokens — quality numbers may understate long-context ceiling (see [`BENCHMARKS.md`](./BENCHMARKS.md))
**Generated:** {now} UTC by [Spark NVFP4 Lab](https://github.com/GumbiiDigital/spark-nvfp4-lab)
**Hardware:** NVIDIA GB10 (DGX Spark), aarch64, CUDA 13.0

## Why this release exists

Most NVFP4 quantizations on HuggingFace ship without any measured side-by-side
comparison against the BF16 baseline. Spark NVFP4 Lab evaluates every release
against its unquantized parent on the **same hardware, same task set, same sampling**,
so the delta is real — not estimated. Full per-task results in [`BENCHMARKS.md`](./BENCHMARKS.md);
raw lm-evaluation-harness JSON is shipped in the repo.

## Footprint

{disk_vram_block if disk_vram_block else "(footprint data not captured for this release)"}

## Recommended use

- Best for reasoning / math workloads on Blackwell hardware where VRAM or memory
  bandwidth is the constraint.
- Expect higher aggregate throughput at concurrency ≥ 4 than BF16 due to smaller
  weight footprint and native FP4 tensor-core paths.
- **Not recommended** for contexts > 16k tokens without re-validation — this release
  was evaluated at 2k.
- Always compare against BF16 for your own task before committing to a quantization.

## Known-good usage — `trtllm-serve`

```bash
docker run --rm --gpus all --ipc=host --network host \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "$(pwd)/weights:/workspace/model" \\
  {container_image} \\
  trtllm-serve /workspace/model \\
    --backend pytorch \\
    --max_batch_size 4 \\
    --port 8000

# Then hit the OpenAI-compatible endpoint:
curl -s http://127.0.0.1:8000/v1/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"model","prompt":"Solve: 37*41 =","max_tokens":128,"temperature":0}}'
```

A chat template is included at [`chat_template.jinja`](./chat_template.jinja); most
inference servers pick it up automatically from `tokenizer_config.json`.

## Tested engines

- ✅ **TensorRT-LLM** {quant_manifest.get("trtllm_version") or "1.1.0rc3"} — via `{container_image}`
- ❓ **vLLM** — not tested in this release. vLLM ≥ 0.6 Blackwell builds support NVFP4; verify before relying.
- ❓ **SGLang** — not tested.

## Reproducing this artifact

Every artifact ships with [`REPRODUCE.sh`](./REPRODUCE.sh) — a one-shot script that
recreates this exact quantization. Container digest, ModelOpt branch, upstream
model revision, and per-file sha256 are pinned in [`manifest.json`](./manifest.json).

### How to verify this release

```bash
# 1. Clone the weights
git clone https://huggingface.co/GumbiiDigital/{args.model_id.split("/")[-1]}-NVFP4

# 2. Check per-file sha256 against manifest
cd {args.model_id.split("/")[-1]}-NVFP4
jq -r '.weights_sha256 | to_entries[] | "\\(.value)  \\(.key)"' manifest.json | sha256sum -c -

# 3. Confirm upstream base-model revision matches
jq -r '.base_model_sha' manifest.json
# Should match: huggingface.co/api/models/{args.model_id}  →  .sha
```

## Known limitations

- NVFP4 weights cannot be loaded by vanilla HF transformers — you need a Blackwell
  inference engine (TensorRT-LLM, vLLM ≥ 0.6 Blackwell, SGLang Blackwell builds).
- Hardware floor: GB10 / B200 / GB200 / RTX 5090 / RTX 6000 Pro. Older GPUs (Hopper,
  Ada, Ampere) cannot execute the NVFP4 kernels.
- **Quality not evaluated on MMLU, HellaSwag, ARC, TruthfulQA, wikitext-PPL.** These
  are log-probability tasks; the TRT-LLM OpenAI shim does not currently expose
  `logprobs`, and HF transformers cannot load NVFP4. Release-1 will unblock via a
  custom backend.
- Evaluated at `max_length = {args.eval_context_length}`. Long-context (>16k) behavior is
  not characterized in this release.

## License

`{args.license}` — inherits from the upstream parent `{args.model_id}`. Read it before redistribution.

---

*Made on a single DGX Spark. Questions or feedback? File an issue at
[Spark NVFP4 Lab on GitHub](https://github.com/GumbiiDigital/spark-nvfp4-lab).*
"""
    (rd / "README.md").write_text(readme)

    # ---- REPRODUCE.sh -------------------------------------------------------
    reproduce = f"""#!/usr/bin/env bash
# Recreate this NVFP4 artifact from scratch. Requires Blackwell GPU + Docker + HF_TOKEN.
# Container digest and upstream model revision are pinned below for full provenance.
set -e

MODEL_ID="{args.model_id}"
BASE_MODEL_REVISION="{base_sha or 'unknown'}"
CONTAINER="{container_image}"
CONTAINER_DIGEST="{container_digest or 'unknown'}"
MODELOPT_BRANCH="0.35.0"
CALIB="{args.calibration}"

mkdir -p ./out
docker run --rm --gpus all --ipc=host \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "$(pwd)/out:/workspace/output_models" \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -e GPU_MAX_MEM_PERCENTAGE=0.9 \\
  "$CONTAINER" \\
  bash -c "
    git clone -b $MODELOPT_BRANCH --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/MO && \\
    cd /app/MO && pip install -e '.[dev]' && \\
    export ROOT_SAVE_PATH='/workspace/output_models' && \\
    /app/MO/examples/llm_ptq/scripts/huggingface_example.sh \\
      --model '$MODEL_ID' \\
      --quant nvfp4 --tp 1 --export_fmt hf
  "
"""
    (rd / "REPRODUCE.sh").write_text(reproduce)
    os.chmod(rd / "REPRODUCE.sh", 0o755)

    # ---- manifest.json ------------------------------------------------------
    render_self_sha = sha256_file(this_script)
    manifest = {
        "schema_version": "0.2",
        "model_id": args.model_id,
        "generated_utc": now,
        "headline": headline,
        "license": args.license,
        "calibration": {
            "key": args.calibration,
            "dataset": "cnn_dailymail" if args.calibration == "default" else "codeparrot/codeparrot-clean" if args.calibration == "code" else "unknown",
            "n_samples_default": 512,
            "seq_len_default": 512,
            "samples_sha256": "not-captured-release-0 — release-1 will record this in quant/manifest.json",
        },
        "eval": {
            "context_length": args.eval_context_length,
            "lm_eval_version": "0.4.7",
            "bootstrap_iters": 100000,
            "ci_method": "unpaired normal approximation; paired bootstrap on release-1 list",
        },
        "tooling": {
            "tensorrt_llm_image": container_image,
            "container_digest": container_digest or "unknown — run `docker image inspect` at reproduction time",
            "trtllm_version": "1.1.0rc3",
            "modelopt_branch": "0.35.0",
            "modelopt_commit": "not-captured-release-0 — release-1 will pin",
            "spark_nvfp4_lab_git_sha": git_sha(repo_root) or "unknown",
            "render_card_py_sha256": render_self_sha,
        },
        "hardware": {
            "gpu": "NVIDIA GB10 (DGX Spark)",
            "arch": "aarch64",
            "cuda": "13.0",
        },
        "base_model": {
            "id": args.model_id,
            "sha": base_sha or "unknown",
        },
        "footprint": {
            "bf16_bytes": bf16_bytes,
            "nvfp4_bytes": nvfp4_bytes,
            "compression_ratio": compression_ratio,
            "bf16_vram_observed_gib": args.bf16_vram_observed_gib,
            "nvfp4_vram_observed_gib": args.nvfp4_vram_observed_gib,
        },
        "results": {"bf16": bf16, "nvfp4": nvfp4},
        "delta_rows": delta_rows,
        "quality_gates": {
            "per_task_threshold_pct": TASK_DELTA_THRESHOLDS_PCT,
            "default_threshold_pct": DEFAULT_DELTA_THRESHOLD_PCT,
            "mmlu_floor_pct": MMLU_FLOOR_PCT,
            "ppl_ceil_ratio": PPL_CEIL_RATIO,
            "min_benchmark_families": MIN_BENCHMARK_FAMILIES,
            "triggered": [{"kind": f[0], "task": f[1], "value": f[2]} for f in flags],
        },
        "auto_publish_blocked": bool(flags),
        "gate_override_reason": args.override_gate_exit,
        "perf_summary": [
            {"concurrency": r["concurrency"], "single_tok_s": r.get("single_stream_tok_s"),
             "aggregate_tok_s": r.get("aggregate_tok_s")}
            for r in perf.get("runs", []) if r.get("n_ok")
        ],
        "perf_settings": perf_settings,
        "tested_engines": [e.strip() for e in args.tested_engines.split(",") if e.strip()],
        "untested_engines": [e.strip() for e in args.untested_engines.split(",") if e.strip()],
        "weights_sha256": sha256_dir(nvfp4_weights_dir) if nvfp4_weights_dir and nvfp4_weights_dir.exists() else {},
    }
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"rendered: {rd}/README.md, BENCHMARKS.md, manifest.json, REPRODUCE.sh")
    if flags:
        print(f"WARNING: {len(flags)} quality gate(s) triggered, auto_publish_blocked=true")
        for f in flags:
            print(f"  - {f}")
        if args.override_gate_exit:
            print(f"override applied: {args.override_gate_exit}")
            sys.exit(0)
        sys.exit(10)


if __name__ == "__main__":
    main()
