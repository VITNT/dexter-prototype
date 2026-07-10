# Dexter Sign Language Recognition

This branch contains the scalable sign-language-recognition redesign work.

The app remains browser-based, but KSL word recognition now has a versioned,
face-free graph and temporal pipeline:

- an architecture diagnosis and migration plan in `docs/sign-language-redesign.md`
- a catalog-first data schema in `data/catalog/sign-catalog.example.json`
- deterministic, tested KSL feature extraction in `src/pipeline/ksl_feature_pipeline.mjs`
- two-hand crop capture plus signer/session-aware dataset recording
- graph-edge spatial features, a causal TCN, focal loss, and optional crop fusion
- signer/session-independent validation with macro-F1 and confusion reporting

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

## KSL v2 Pipeline

New KSL recordings use `ksl-landmarks-v2`:

```text
60 video frames
  -> shoulders/elbows/wrists/hips + left/right hands (150 raw values)
  -> shoulder normalization + velocity + acceleration + 62 graph relations
  -> 700 model features per frame
  -> spatial graph projection + causal TCN
  -> optional left/right grayscale hand-crop fusion
  -> KSL class prediction
```

The recorder requires signer and session IDs. The KSL trainer refuses a
recording-only validation split because clips from the same session can make
validation accuracy look perfect without generalizing.

The historical `csl_*` filenames and internal `csl_word` mode remain in place
for compatibility with existing Cambodian/Khmer assets. The UI calls this KSL.

After training, place the four downloaded files at the project root:

- `csl_model.json`
- `csl_model.weights.bin`
- `csl_labels.json`
- `csl_model_metadata.json`

The live predictor automatically recognizes the new `[60,700]` model and its
optional `[60,24,48,1]` hand-crop input. Existing 30-frame LSTM sub-models stay
available only as a fallback when the unified graph model is absent.

The crop branch is a compact browser-trainable appearance stream, not a
pretrained WLASL video backbone. Optional ASL word files can be installed as
`asl_word_model.json`, `asl_word_model.weights.bin`, `asl_word_labels.json`, and
`asl_word_model_metadata.json`; without them, ASL Words reports that no word
model is installed instead of accidentally using the KSL sub-model registry.

## Verification

Run the deterministic feature tests with:

```bash
node src/pipeline/ksl_feature_pipeline.test.mjs
node src/pipeline/trainer_feature_compat.test.mjs
node src/pipeline/ui_contract.test.mjs
```
