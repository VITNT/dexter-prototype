# External KSL resource compatibility

Source reviewed:

<https://github.com/Phal-Sovandy/Khmer-Sign-Language-Translation-System>

## What is actually published

- 12 WebP collage/presentation images under `src/assets/images/dataset/`
- 41 static form JPGs under `backend/sample/`
- 18 single example videos under `src/assets/videos/sample_signs/`
- PyTorch static LandmarkNet checkpoints for 20 or 42 form classes
- Keras weights for a 75-class, 30-frame, two-hand LSTM

The repository documentation describes larger datasets (22,000+ static
samples and 1,900 temporal videos), but those raw datasets are not present in
the repository. The report also documents a random 90/10 static-image split;
it does not provide signer identities for an independent split.

## What Dexter uses

Dexter converts the released 75-class Keras LSTM into a TensorFlow.js model and
uses it as the default whole-word recognizer when no project-trained KSL v2
model is installed. `tools/export_ksl_dynamic_topology.mjs` and
`tools/export_ksl_dynamic_model.py` perform the deterministic conversion. The
runtime adapter reproduces the source preprocessing: 30 frames, two hands in
MediaPipe detection order, and wrist-relative XYZ values.

The released 42-class static checkpoint remains a browser-side fallback.
`tools/export_ksl_static_model.py` performs its deterministic weights-only
export. The matching released JPGs are used as dictionary references. The
source is ISC licensed; attribution is in `THIRD_PARTY_NOTICES.md`.

## Why these files are not used to retrain KSL v2

KSL v2 expects 60-frame clips containing upper-body pose landmarks and both
hands, with signer or session identifiers. The upstream temporal model expects
30 frames containing only two hands. Static JPGs contain no motion, while one
example video per label cannot provide a signer-independent training and
validation split. Repeating or heavily augmenting those samples would create
leakage and misleading validation accuracy.

To retrain KSL v2, obtain or record at least two independent signers or
sessions for every label, with multiple clips per label in each group. Import
those recordings through `recorder.html`, then train them through
`trainer.html`.
