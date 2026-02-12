# Contributing to pymde

## Project Structure

- `pymde/` — source code (includes a Cython extension at `pymde/preprocess/_graph.pyx`)
- `pyproject.toml` — all project metadata, dependencies, and tool configuration
- `setup.py` — minimal file defining only the Cython extension module
- `scripts/release.sh` — interactive release script

## Development Setup

```bash
# Clone the repo
git clone https://github.com/cvxgrp/pymde.git
cd pymde

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v pymde/
```

## Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting (configured in `pyproject.toml`):

```bash
# Check for lint errors
ruff check pymde/

# Auto-fix lint errors
ruff check --fix pymde/

# Check formatting
ruff format --check pymde/

# Apply formatting
ruff format pymde/
```

## Making a Release

Releases are handled by `scripts/release.sh`, which bumps the version in `pymde/__init__.py`, commits, tags, and pushes. The tag push triggers the CI release workflow that builds wheels and publishes to PyPI.

```bash
./scripts/release.sh patch   # 0.2.3 -> 0.2.4
./scripts/release.sh minor   # 0.2.3 -> 0.3.0
```
