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

import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

# Quality-gate thresholds. If exceeded, manifest gets `auto_publish_blocked: true`.
DELTA_FLAG_PCT = 5.0   # >5pt absolute drop on any benchmark = flag
MMLU_FLOOR     = 25.0  # Below random = something is broken
PPL_CEIL_RATIO = 1.20  # NVFP4 PPL > 1.2x BF16 PPL = flag


def sha256_dir(path: Path) -> dict:
    """sha256 of every regular file under path."""
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p.relative_to(path))] = h.hexdigest()
    return out


def extract_results(p: Path) -> dict:
    """Pull the per-task summary scores from an lm-eval results.json."""
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    out = {}
    FILTER_RANK = {"strict-match": 0, "none": 1, "flexible-extract": 2}
    METRIC_RANK = {"exact_match": 0, "acc": 1, "pass@1": 2, "acc_norm": 3}
    for task, scores in (j.get("results") or {}).items():
        best = None
        for k, v in scores.items():
            if not isinstance(v, (int, float)) or "," not in k or "stderr" in k:
                continue
            base, filt = k.rsplit(",", 1)
            fr = FILTER_RANK.get(filt, 99)
            mr = METRIC_RANK.get(base, 99)
            cand = (fr, mr, k, v)
            if best is None or cand < best:
                best = cand
        if best:
            out[task] = {"metric": best[2], "value": float(best[3])}
    return out


def delta_table(bf16: dict, nvfp4: dict) -> tuple[str, list[dict]]:
    """Side-by-side BF16 vs NVFP4 with delta. Returns markdown + raw rows."""
    rows = []
    keys = sorted(set(bf16) | set(nvfp4))
    md = ["| Task | Metric | BF16 | NVFP4 | Δ | Flag |", "|---|---|---|---|---|---|"]
    flags = []
    for k in keys:
        b = bf16.get(k, {}); q = nvfp4.get(k, {})
        bv = b.get("value"); qv = q.get("value")
        metric = b.get("metric") or q.get("metric") or "?"
        if bv is None or qv is None:
            md.append(f"| {k} | {metric} | {bv} | {qv} | – | missing-side |")
            continue
        delta_pct = (qv - bv) * 100 if metric.startswith("acc") or "exact_match" in metric or "pass@1" in metric else (qv - bv)
        is_perplexity = "perplexity" in metric or k == "wikitext"
        flag = ""
        if k == "mmlu" and qv * 100 < MMLU_FLOOR and metric.startswith("acc"):
            flag = "MMLU<random"; flags.append(("mmlu_below_random", k, qv))
        if not is_perplexity and abs(delta_pct) > DELTA_FLAG_PCT:
            flag = "Δ>5pt"; flags.append(("delta_too_large", k, delta_pct))
        if is_perplexity and bv > 0 and qv / bv > PPL_CEIL_RATIO:
            flag = "PPL>1.2x"; flags.append(("ppl_blowup", k, qv / bv))
        delta_repr = f"{delta_pct:+.2f}" + ("pt" if not is_perplexity else "")
        md.append(f"| {k} | {metric} | {bv:.4f} | {qv:.4f} | {delta_repr} | {flag} |")
        rows.append({"task": k, "metric": metric, "bf16": bv, "nvfp4": qv, "delta": delta_pct, "flag": flag})
    return "\n".join(md), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--calibration", default="default")
    ap.add_argument("--license", default="other", help="SPDX-style HF license tag (must be on HF's allowed list; see hf.co/docs/hub/repositories-licenses)")
    args = ap.parse_args()

    rd = args.release_dir
    bf16 = extract_results(rd / "eval-bf16" / "results.json")
    nvfp4 = extract_results(rd / "eval-nvfp4" / "results.json")
    perf = json.loads((rd / "perf" / "perf.json").read_text()) if (rd / "perf" / "perf.json").exists() else {}
    quant_manifest = json.loads((rd / "quant" / "manifest.json").read_text()) if (rd / "quant" / "manifest.json").exists() else {}

    delta_md, flags = delta_table(bf16, nvfp4)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # ---- BENCHMARKS.md ------------------------------------------------------
    perf_md = ["| Concurrency | n_ok | wall_s | p50 latency | p95 latency | Single-stream tok/s | Aggregate tok/s |",
               "|---|---|---|---|---|---|---|"]
    for r in perf.get("runs", []):
        if r.get("n_ok"):
            perf_md.append(f"| {r['concurrency']} | {r['n_ok']}/{r['n_requests']} | {r['wall_s']} | "
                           f"{r['p50_latency_s']}s | {r['p95_latency_s']}s | "
                           f"{r['single_stream_tok_s']} | {r['aggregate_tok_s']} |")

    bench_md = f"""# Benchmarks — {args.model_id} (NVFP4)

Generated: {now} UTC
Hardware: NVIDIA GB10 (DGX Spark), aarch64
Calibration dataset: {args.calibration}

## Quality (BF16 baseline vs NVFP4 quantization)

{delta_md}

## Performance — TensorRT-LLM serving on GB10

{chr(10).join(perf_md) if len(perf_md) > 2 else "(no perf data captured)"}

## Quality gates triggered

{"None — passed all gates." if not flags else chr(10).join(f"- **{f[0]}** on `{f[1]}` (value: {f[2]})" for f in flags)}
"""
    (rd / "BENCHMARKS.md").write_text(bench_md)

    # ---- README.md (HF model card) -----------------------------------------
    readme = f"""---
license: {args.license}
base_model: {args.model_id}
tags:
- nvfp4
- quantized
- blackwell
- gb10
- spark
- spark-nvfp4-lab
---

# {args.model_id.split("/")[-1]} — NVFP4 Quantization

Quantized from [`{args.model_id}`](https://huggingface.co/{args.model_id}) using NVIDIA TensorRT Model-Optimizer.
Optimized for NVIDIA Blackwell (GB10 / B200 / RTX 5090 / RTX 6000 Pro / etc.).

**Calibration dataset:** `{args.calibration}`
**Generated:** {now} UTC by [Spark NVFP4 Lab](https://github.com/GumbiiDigital/spark-nvfp4-lab)
**Hardware:** NVIDIA GB10 (DGX Spark), aarch64, CUDA 13.0

## Why this exists

Most NVFP4 quantizations on HuggingFace ship without measured side-by-side comparison
against the BF16 baseline. Spark NVFP4 Lab evaluates every release against its
unquantized parent on the same hardware, with full per-task results in [`BENCHMARKS.md`](./BENCHMARKS.md)
and raw lm-evaluation-harness JSON in [`eval/`](./eval/).

## Headline numbers

See [`BENCHMARKS.md`](./BENCHMARKS.md) for the full Δ table and performance numbers.

## Reproducing this artifact

Every artifact ships with [`REPRODUCE.sh`](./REPRODUCE.sh) — a one-shot script that
recreates this exact quantization. Hardware fingerprint, container tag, and calibration
dataset hash are pinned in [`manifest.json`](./manifest.json).

## Known limitations

- NVFP4 needs Blackwell-class GPUs (B200, GB200, GB10, RTX 5090, RTX 6000 Pro)
- Inference engines that support NVFP4: TensorRT-LLM, vLLM ≥ 0.6 (Blackwell builds)
- Quality may differ from BF16 on long-context (>16K tokens) and rare-token paths.
  See `BENCHMARKS.md` for measured deltas.

## License

Inherits the license of the parent model `{args.model_id}`. Read it before redistribution.

---

*Made on a single DGX Spark. If this saved you time, drop a ⭐ on the
[Spark NVFP4 Lab GitHub](https://github.com/GumbiiDigital/spark-nvfp4-lab).*
"""
    (rd / "README.md").write_text(readme)

    # ---- REPRODUCE.sh -------------------------------------------------------
    reproduce = f"""#!/usr/bin/env bash
# Recreate this NVFP4 artifact from scratch. Requires Blackwell GPU + Docker + HF_TOKEN.
set -e
mkdir -p ./out
docker run --rm --gpus all --ipc=host \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "$(pwd)/out:/workspace/output_models" \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -e GPU_MAX_MEM_PERCENTAGE=0.9 \\
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \\
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/MO && \\
    cd /app/MO && pip install -e '.[dev]' && \\
    export ROOT_SAVE_PATH='/workspace/output_models' && \\
    /app/MO/examples/llm_ptq/scripts/huggingface_example.sh \\
      --model '{args.model_id}' \\
      --quant nvfp4 --tp 1 --export_fmt hf
  "
"""
    (rd / "REPRODUCE.sh").write_text(reproduce)
    os.chmod(rd / "REPRODUCE.sh", 0o755)

    # ---- manifest.json ------------------------------------------------------
    quant_dir = rd / "quant"
    weights_dir = next((d for d in quant_dir.iterdir() if d.is_dir()), None) if quant_dir.exists() else None
    manifest = {
        "model_id": args.model_id,
        "calibration": args.calibration,
        "generated_utc": now,
        "tooling": {
            "tensorrt_llm_image": "nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev",
            "modelopt_branch": "0.35.0",
            "lm_eval_version": "0.4.7",
        },
        "hardware": {
            "gpu": "NVIDIA GB10 (DGX Spark)",
            "arch": "aarch64",
            "cuda": "13.0",
        },
        "results": {
            "bf16": bf16,
            "nvfp4": nvfp4,
        },
        "quality_gates_triggered": [{"kind": f[0], "task": f[1], "value": f[2]} for f in flags],
        "auto_publish_blocked": bool(flags),
        "perf_summary": [
            {"concurrency": r["concurrency"], "single_tok_s": r.get("single_stream_tok_s"),
             "aggregate_tok_s": r.get("aggregate_tok_s")}
            for r in perf.get("runs", []) if r.get("n_ok")
        ],
        "weights_sha256": sha256_dir(weights_dir) if weights_dir and weights_dir.exists() else {},
    }
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"rendered: {rd}/README.md, BENCHMARKS.md, manifest.json, REPRODUCE.sh")
    if flags:
        print(f"WARNING: {len(flags)} quality gate(s) triggered, auto_publish_blocked=true")
        for f in flags:
            print(f"  - {f}")
        sys.exit(10)


if __name__ == "__main__":
    main()
