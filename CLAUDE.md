# Seldon

## Setup
- Use `uv` for package management, not pip
- Virtual environment: `.venv/`
- Install: `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`

## Project structure
- `seldon/` — main package
- NumPyro (JAX) for probabilistic modeling
- Pydantic for data models
