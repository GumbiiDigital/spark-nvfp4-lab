# Spark NVFP4 Lab

Local GPU inference experiments for NVFP4 quantization and Spark model workflows.

## Status

Experimental lab repo. This is a working area for local inference notes, scripts, and reproducible checks rather than a polished product.

## Why this exists

The goal is to understand how far local, user-owned hardware can go for practical AI workloads without depending on hosted model APIs.

## Focus

- NVFP4 / low-precision inference experiments
- Local GPU workflow notes
- Repeatable setup and verification scripts
- Practical model-running constraints: memory, speed, quality, and reliability

## Principles

- Local-first by default
- Reproducible commands over vague notes
- Clear separation between experiments and production-ready claims
- Document limitations honestly

## Suggested structure

```text
scripts/      runnable checks and setup helpers
notes/        experiment notes and results
docs/         longer writeups or diagrams
```

## Privacy

This repo should not require API keys or hosted model services for the core local-inference experiments. Any optional external dependency should be documented explicitly.

## Next cleanup targets

- Add exact hardware/software baseline
- Add one-command environment check
- Add benchmark notes in a consistent table
- Add screenshots or terminal output from successful runs
