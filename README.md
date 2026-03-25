# TurboQuant

TurboQuant is a reference implementation of low-bit transformer KV-cache compression based on the TurboQuant paper. This repository includes PyTorch reference kernels, an end-to-end demo, benchmark notes, and an early vLLM plugin scaffold.

## Status

- PyTorch encode/decode and attention paths are implemented.
- The reference attention kernel uses a QJL correction term with `q @ S`.
- Triton and LUT-based acceleration paths are still experimental.
- vLLM integration is present as a plugin scaffold and needs broader validation.

## What Is Here

- `src/`: reference kernels, cache implementation, demo, and LUT experiments
- `vllm_plugin/`: plugin package skeleton for vLLM integration
- `BENCHMARKS.md`: current measurements and quality notes
- `deploy/`: example deployment assets for serving with vLLM

## Quick Start

Install the package from the repository root:

```bash
pip install -e .
```

Run the demo:

```bash
python src/demo.py
```

## Implementation Summary

TurboQuant combines:

1. PolarQuant-style low-bit reconstruction for the main KV signal.
2. A QJL sign sketch for correcting inner products during attention scoring.
3. Norm side-information so keys and values can be reconstructed or scored efficiently.

At `d=128`, the current format stores about `52` bytes per vector, or roughly `3.25` bits per value including metadata.

## vLLM

The package exports a vLLM platform plugin entry point. After installation in an environment with vLLM available, the intended serving flow is:

```bash
vllm serve <model> --attention-backend turboquant
```

The plugin path is still under active development.

## License

MIT
