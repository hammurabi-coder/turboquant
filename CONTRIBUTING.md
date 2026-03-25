# Contributing to TurboQuant

Thanks for your interest in contributing! TurboQuant is the first open-source implementation of the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Getting Started

```bash
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e ".[dev]"
pytest src/test_turboquant.py -v
```

## What We Need Help With

- **Benchmarking on more models** — We've validated on Mistral-7B and Nemotron-Nano-4B. More models = better.
- **Triton kernel correctness** — `kernels.py` is experimental and uses Rademacher S matrices (see IMPLEMENTATION_NOTES.md). Needs validation.
- **vLLM integration** — The plugin scaffold in `vllm_plugin/` needs real-world testing.
- **Performance optimization** — The pure PyTorch path is correct but slow. GPU acceleration welcome.
- **More bit-width configurations** — The paper shows results at 2.5-bit and 3.5-bit. We support both but need more testing.

## Code Style

- Python 3.10+
- Type hints on public API functions
- Docstrings on all public classes and functions
- `pytest` for tests — run `pytest src/test_turboquant.py -v` before submitting

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass: `pytest src/test_turboquant.py -v`
4. Update documentation if needed
5. Open a PR with a clear description of what changed and why

## Reporting Issues

Please include:
- Python and PyTorch versions
- GPU model (if applicable)
- Minimal reproduction code
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
