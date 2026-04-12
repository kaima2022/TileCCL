# Contributing to TileCCL

Thank you for your interest in TileCCL. Contributions of all kinds are welcome: bug reports, feature proposals, documentation improvements, and code.

## Development Setup

```bash
git clone https://github.com/MaKai-Research/tileccl.git
cd tileccl
pip install -e ".[dev]"
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

## Pull Requests

1. Fork the repository and create a feature branch.
2. Run `make lint` and `make typecheck` before submitting.
3. Keep PRs focused: one feature or fix per PR.

## Reporting Issues

When reporting a bug, please include:

- Hardware (GPU model, interconnect type)
- Software versions (Python, PyTorch, Triton, CUDA)
- Minimal reproduction script
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
