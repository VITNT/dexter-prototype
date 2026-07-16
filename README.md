# Dexter Sign Language Recognition

For a fresh clone, follow [SETUP.md](SETUP.md). VS Code **Go Live** alone is
not enough for ASL Words because WLASL300 runs in a local Python service and its
checkpoint is excluded from Git.

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

## PhonSSM WLASL300 service

The isolated ASL-word backend lives in `asl_service/` and uses the pose+hands
PhonSSM architecture from
<https://github.com/anonymoussubmitter-167/phon-ssm>. The browser sends the
MediaPipe pose and hand landmarks it already tracks, so no RGB I3D model or
second server-side landmark detector is required. Checkpoint installation
details are in `asl_service/models/README.md`.

After installing the checkpoint and Python requirements, run:

```powershell
.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Check `http://127.0.0.1:8001/health`. It should report `"status":"ready"`.
The live ASL-word client uses a rolling 2.5-second sign window. Each window is
uniformly sampled to 30 frames of 33 pose, 21 left-hand, and 21 right-hand
landmarks, then shoulder-centered and shoulder-width-scaled exactly like the
reference WLASL training pipeline.
Because WLASL has no idle class, motion, confidence, and top-two-margin gates
reject no-sign windows instead of forcing an arbitrary vocabulary word.
Pausing or resuming the camera clears all cached recognition frames.

The linked PhonSSM repository does not publish its WLASL300 `best_model.pt`—the
benchmark weight is excluded by its `*.pt` ignore rule. The service is therefore
ready for a matching checkpoint, but cannot report `"ready"` until that weight
is obtained from the authors or reproduced with their benchmark training script.

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

When no project-trained KSL v2 bundle is installed, the translator loads the
released 75-class whole-word model from `models/ksl/legacy-word-v1/`. It uses
30 frames of wrist-relative two-hand landmarks and appends the recognized Khmer
word directly to the sentence box. A released 42-class static LandmarkNet is a
final fallback for held hand forms. Its matching reference images appear in the
KSL dictionary. Source and license details are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The linked upstream repository does not publish the raw 22,000-image or
1,900-video datasets described by its documentation. Its 12 dataset collage
images, 41 form JPGs, and 18 single example videos are not sufficient to train
or independently validate the KSL v2 temporal model. See
[`docs/ksl-external-resources.md`](docs/ksl-external-resources.md) for the
compatibility audit and the additional data needed for temporal retraining.

The crop branch is a compact browser-trainable appearance stream, not the ASL
word backbone. The disabled RGB-video bundle under `models/asl/wlasl/` remains
only as an optional legacy browser path; the active backend contract is
documented in `docs/wlasl-integration.md`.
Optional legacy ASL word files can still be installed as
`asl_word_model.json`, `asl_word_model.weights.bin`, `asl_word_labels.json`, and
`asl_word_model_metadata.json`; without them, ASL Words reports that no word
model is installed instead of accidentally using the KSL sub-model registry.

## Verification

Run the deterministic feature tests with:

```bash
node src/pipeline/ksl_feature_pipeline.test.mjs
node src/pipeline/ksl_legacy_word_pipeline.test.mjs
node src/pipeline/ksl_legacy_model_asset.test.mjs
node src/pipeline/ksl_static_classifier.test.mjs
node src/pipeline/trainer_feature_compat.test.mjs
node src/pipeline/ui_contract.test.mjs
node src/pipeline/wlasl_video_pipeline.test.mjs
```
