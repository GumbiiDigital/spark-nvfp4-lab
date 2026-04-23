# Spark NVFP4 Lab

Quantize any HuggingFace model to **NVFP4**, evaluate it against its BF16 baseline on the same hardware, publish a model card with real numbers — not guesses.

Built for a single DGX Spark (GB10 Blackwell). Works on any Blackwell-class GPU.

## Releases

| Model | HF link | gsm8k Δ (BF16 → NVFP4) |
|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | [GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4](https://huggingface.co/GumbiiDigital/DeepSeek-R1-Distill-Llama-8B-NVFP4) | −3.94 pt |

## Why

Most NVFP4 quants on HuggingFace ship without any side-by-side comparison against the unquantized parent. You can't tell whether the quant cost you 1 point, 10 points, or destroyed the model. This lab runs both halves on the same hardware, same task set, same sampling — so the delta is real.

## Requirements

- **Hardware:** Blackwell-class GPU (GB10, B200, GB200, RTX 5090, RTX 6000 Pro, …)
- **Software:** CUDA 13.0+, Docker with NVIDIA Container Toolkit
- **Container:** `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`
- **Account:** HuggingFace token (read for base model, write for publishing) — `hf auth login`

## Quick start

```bash
git clone https://github.com/GumbiiDigital/spark-nvfp4-lab.git
cd spark-nvfp4-lab

# Full pipeline: quantize → serve → eval NVFP4 → perf → eval BF16 baseline
scripts/pipeline.sh --release release-0_MyModel --model <hf-model-id>

# Render the model card
python3 scripts/render_card.py \
  --release-dir results/release-0_MyModel \
  --model-id <hf-model-id> \
  --license mit
```

See [`SOP.md`](./SOP.md) for the full operational reference — pre-flight checklist, incident response, known infrastructure constraints, release readiness checklist.

## Known constraints

- `trtllm-serve` OpenAI shim does not expose logprobs, so MMLU / HellaSwag / wikitext PPL can't run on NVFP4 weights via this path today. Generation-only tasks (gsm8k, BBH generation, etc.) work.
- NVFP4 requires Blackwell architecture. Older hardware can't load the weights.

## Layout

```
scripts/          # pipeline.sh, quantize.sh, serve.sh, eval.sh, perf.sh, render_card.py, _lib.sh
SOP.md            # operational procedures + lessons learned
results/          # per-release artifacts (gitignored — weights live here)
models/           # materialized HF snapshots (gitignored)
```

## License

MIT for the scripts in this repo. Generated model cards inherit the license of the upstream model being quantized — always check before redistributing.
