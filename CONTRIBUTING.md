# Contributing to TNCC

Thank you for your interest in TNCC. Contributions of all kinds are welcome: bug reports, feature proposals, documentation improvements, and code.

## Development Setup

```bash
git clone https://github.com/MaKai-Research/tncc.git
cd tncc
pip install -e ".[dev,benchmark]"
```

## Code Style

- **Linter/formatter:** [Ruff](https://docs.astral.sh/ruff/) (line length 100)
- **Type checker:** mypy (strict on public API)

```bash
make lint       # Check
make lint-fix   # Auto-fix
make format     # Format
make typecheck  # Type check
```

## Testing

TNCC tests require **2 NVIDIA GPUs with NVLink** for the full suite.

```bash
make test          # Full suite (requires 2x GPUs)
make test-unit     # CPU-only unit tests (no GPU required)
make test-multigpu # Multi-GPU tests only
```

## Pull Requests

1. Fork the repository and create a feature branch.
2. Write tests for new functionality.
3. Run `make lint` and `make test-unit` before submitting.
4. Keep PRs focused: one feature or fix per PR.

## Reporting Issues

When reporting a bug, please include:

- Hardware (GPU model, interconnect type)
- Software versions (Python, PyTorch, Triton, CUDA)
- Minimal reproduction script
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
