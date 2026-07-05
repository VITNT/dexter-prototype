# Dexter Sign Language Recognition

This branch contains the scalable sign-language-recognition redesign work.

The current app remains a browser-based prototype, but the redesign adds:

- an architecture diagnosis and migration plan in `docs/sign-language-redesign.md`
- a catalog-first data schema in `data/catalog/sign-catalog.example.json`
- starter sequence pipeline modules in `src/pipeline/`

## Run Locally

Serve the project from the repository root so browser `fetch()` calls can load
models, labels, and JSON files:

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/index.html` for the current translator prototype
- `http://localhost:8000/recorder.html` for the dataset recorder prototype
- `http://localhost:8000/trainer.html` for the browser trainer prototype

## Redesign Direction

The new architecture should process a recorded sign clip as a sequence:

```text
recorded video -> sampled frames -> MediaPipe keypoints -> normalized sequence -> model -> translation
```

Start with isolated sign recognition using MediaPipe keypoints plus an LSTM,
GRU, or Temporal CNN. Add continuous sentence recognition only after the
dataset and isolated-sign pipeline are stable.
