# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r docs/requirements.txt
```

## Documentation workflow

1. Edit Markdown or reStructuredText files under `docs/source`.
2. Build locally with Sphinx.
3. Fix warnings before opening a pull request.

```bash
python -m sphinx -b html -W docs/source docs/_build/html
```

## Style guidance

- Prefer runnable command examples with placeholder paths under `data/` or `runs/`.
- Document HDF5 dataset names explicitly when adding or changing loaders.
- Keep API docs generated from docstrings, and put workflow explanations in the user guide.
- Avoid committing generated docs in `docs/_build/`.
