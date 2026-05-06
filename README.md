# Spark NVFP4 Lab

<!-- GUMBII-DOCS:START -->
## Repository Status

- **Repository:** `GumbiiDigital/spark-nvfp4-lab`
- **Title:** spark-nvfp4-lab
- **Classification:** Active product (`active-product`)
- **Visibility:** public
- **Default branch:** `main`
- **Summary:** Spark NVFP4 Lab
- **Primary contents:** Root folders: `.github`, `scripts`, `tests` Key root files: `.gitignore`, `LICENSE`, `SOP.md`
- **Stack/signals:** Shell, Python, Python
- **Topics:** `audit-high`, `gumbii`, `python`, `active-product`
- **Recommended action:** Keep active; add README/status notes if missing.
- **Documentation refreshed:** 2026-05-06

This section is maintained by GumbiiDigital repository hygiene tooling. Preserve the markers when editing.
<!-- GUMBII-DOCS:END -->


[![CI](https://github.com/GumbiiDigital/spark-nvfp4-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/GumbiiDigital/spark-nvfp4-lab/actions/workflows/ci.yml)

Quantize any HuggingFace model to **NVFP4**, evaluate it against its BF16 baseline on the same hardware, publish a model card with real numbers — not guesses.

Built for a single DGX Spark (GB10 Blackwell). Works on any Blackwell-class GPU.

## Releases

| Model | HF link | gsm8k Δ (BF16 → NVFP4) | Compression |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | [GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4](https://huggingface.co/GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4) | −3.94 pp (95% CI ±3.69 pp) | 2.66× on disk, 1.93× VRAM |

## Why

Most NVFP4 quants on HuggingFace ship without any side-by-side comparison against the unquantized parent. You can't tell whether the quant cost you 1 point, 10 points, or destroyed the model. This lab runs both halves on the same hardware, same task set, same sampling — so the delta is real.

## Requirements

- **Hardware:** Blackwell-class GPU (GB10, B200, GB200, RTX 5090, RTX 6000 Pro, …)
- **Software:** CUDA 13.0+, Docker with NVIDIA Container Toolkit
- **Container:** `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`
- **Account:** HuggingFace token (read for base model, write for publishing) — `hf auth login`

## Quick start — quantize your own model

```bash
git clone https://github.com/GumbiiDigital/spark-nvfp4-lab.git
cd spark-nvfp4-lab

# Full pipeline: quantize → serve → eval NVFP4 → perf → eval BF16 baseline
scripts/pipeline.sh --release release-N_MyModel --model <hf-model-id>

# Render the model card
python3 scripts/render_card.py \
  --release-dir results/release-N_MyModel \
  --model-id <hf-model-id> \
  --license mit \
  --bf16-weights-dir models/MyModel-bf16 \
  --bf16-vram-observed-gib <value> \
  --nvfp4-vram-observed-gib <value>
```

## Quick start — run an existing release

```bash
# One-shot: pull weights + serve on port 8000
docker run --rm --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4 \
    --backend pytorch --max_batch_size 4 --port 8000

# Then hit the OpenAI-compatible endpoint:
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"Solve: 37*41 =","max_tokens":128,"temperature":0}'
```

See [`SOP.md`](./SOP.md) for the full operational reference — pre-flight checklist, incident response, known infrastructure constraints, release readiness checklist, and changelog.

## Known constraints (release-0 as of 2026-04-23)

- `trtllm-serve`'s OpenAI shim does not expose `logprobs`, so MMLU / HellaSwag / ARC / wikitext-PPL cannot currently be evaluated on NVFP4 weights through this path. Generation-only tasks (gsm8k, BBH generation, MATH-500, etc.) work. Unblocking this is on the release-1 list.
- NVFP4 requires Blackwell architecture. Older hardware can't load the weights.
- Eval context length is capped at 2048 in release-0; long-context ceiling is not characterized.

## Layout

```
scripts/          # pipeline.sh, quantize.sh, serve.sh, eval.sh, perf.sh, render_card.py, _lib.sh
tests/            # golden fixture + functional test for render_card.py
.github/workflows # CI: shellcheck + render_card test
SOP.md            # operational procedures, incident response, changelog
results/          # per-release artifacts (gitignored — weights live here)
models/           # materialized HF snapshots (gitignored)
```

## Contributing

Issues and PRs welcome. Before opening a PR: `bash tests/test_render_card.sh` must pass and `shellcheck -S warning scripts/*.sh tests/*.sh` must be clean — both run automatically in CI.

## License

[MIT](./LICENSE) for the scripts in this repo. Generated model cards inherit the license of the upstream model being quantized — always verify the base model's license before redistributing any quantized artifact.
