# AGENTS.md

## Environment

- Python 3.13 required (managed by `uv`)
- Dependencies: matplotlib, numpy, opencv-python
- Virtual environment: `.venv/` (managed by `uv`)

## Commands

```bash
# Install dependencies
uv sync

# Run a homework script
uv run python src/hw1/5a_gaussian_noise.py
# Or activate venv first:
.venv\Scripts\activate && python src/hw1/5a_gaussian_noise.py
```

## Structure

- `src/hw1/`, `src/hw2/`: Homework scripts (image processing / signal processing)
- `resource/HW1_img/`, `resource/HW2_img/`: Input images referenced by scripts
- `doc/`: Assignment PDFs
- Scripts load images via relative paths like `./resource/HW1_img/...` (run from repo root)

## Conventions

- Scripts are standalone and display results with `plt.show()`
- Code comments are in Chinese (assignment language)
- No tests, no linting, no CI — this is a coursework repo
- `main.py` is a uv template placeholder, not the real entrypoint
