# Contributing

Thanks for your interest in this project. It is research code, so ideas, critique and negative results are as valuable as code.

## Ways to help

- **Discuss:** open an issue with questions, related papers, or objections to the design.
- **Experiment:** propose or run an ablation. Use the protocol in [docs/experiments.md](docs/experiments.md) so results are comparable.
- **Code:** fix bugs, improve performance (for example, a fused CfC kernel), or add baselines.

## Reporting results

When you share a result, include:

- the config file and git commit,
- dataset and split,
- parameter count,
- the metrics defined in the protocol, plus audio samples,
- the GPU used and the training time.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each PR focused on one change.
3. Add or update tests and docs where relevant.
4. Describe what changed and why, with numbers for anything that affects model quality or speed.

## Code style

Python 3.10+, formatted and linted with `ruff`, with type hints on public functions. Tooling will be set up together with the first code.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
