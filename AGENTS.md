# Repository Guidelines

## Project Structure & Module Organization

```
ceti-whale-sounds/
├── src/                     # Core analysis modules
│   ├── click_detector.py    # Energy/envelope click detection
│   ├── coda_detector.py     # Group clicks into codas
│   └── feature_extractor.py # Rhythm/tempo/rubato/ornamentation
├── app.py                   # Streamlit UI
├── scripts/
│   └── download_sample_data.py
├── data/raw/watkins/        # Sample audio (downloaded; gitignored)
├── test_improvements.py     # Scripted checks of fixes
├── test_parameter_fixes.py  # Parameter passing tests
├── test_real_whale_data.py  # Real-data analysis (requires samples)
├── pyproject.toml           # Python 3.11+, deps
└── requirements.txt         # Locked deps (uv export)
```

## Build, Test, and Development Commands

- Setup env (uv): `uv venv && source .venv/bin/activate && uv sync`
- Setup env (pip): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Download data: `python scripts/download_sample_data.py`
- Run UI: `streamlit run app.py` (open http://localhost:8501)
- Quick tests (scripted):
  - `python test_parameter_fixes.py`
  - `python test_improvements.py`
  - `python test_real_whale_data.py` (after downloading samples)
- Optional pytest (if installed): `pytest -q` (root-level `test_*.py` files)

## Coding Style & Naming Conventions

- PEP 8, 4-space indentation, type hints encouraged.
- Modules: `snake_case.py` under `src/`; tests: `test_*.py` at repo root.
- Classes: `PascalCase`; functions/variables: `snake_case`.
- Docstrings: triple-quoted, include purpose, params, returns, and scientific context when relevant.
- Keep pure logic in `src/`; user I/O and UI in `app.py`/`scripts/`.

## Testing Guidelines

- Prefer fast, deterministic tests; avoid network in tests.
- Real-data tests depend on files in `data/raw/watkins/` (download via script).
- Add assertions for new behavior; keep output stable for comparison.
- Optional coverage (if available): `pytest --cov=src`.

## Commit & Pull Request Guidelines

- Commits: imperative, concise, scoped (e.g., "fix(click_detector): correct envelope padding").
- Reference area(s): `click_detector`, `coda_detector`, `feature_extractor`, `app`.
- Update docs when behavior changes (README, CLAUDE.md) and add CHANGELOG entry for user-visible changes.
- PRs must include:
  - Clear description, rationale, and before/after notes (scientific impact when applicable).
  - Steps to test; screenshots/GIFs for UI changes.
  - New/updated tests and data handling notes.

## Security & Data Tips

- Do not commit audio files; large media is gitignored. Use `scripts/download_sample_data.py`.
- Validate inputs and sanitize paths; avoid implicit network calls at runtime.
- Be transparent about detection parameters and sample-rate assumptions in code and UI.

