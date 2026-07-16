# External CSL resource compatibility

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
uses it as the CSL whole-word recognizer. `tools/export_csl_dynamic_topology.mjs` and
`tools/export_csl_dynamic_model.py` perform the deterministic conversion. The
runtime adapter reproduces the source preprocessing: 30 frames, two hands in
MediaPipe detection order, and wrist-relative XYZ values.

The released 42-class static checkpoint remains a browser-side fallback.
`tools/export_csl_static_model.py` performs its deterministic weights-only
export. Six approved released JPGs remain as dictionary references; other
cards intentionally show an unavailable-image placeholder. The source is ISC
licensed; attribution is in `THIRD_PARTY_NOTICES.md`.

## Why this repository does not retrain the model

The upstream temporal model expects 30-frame two-hand clips, but its training
videos and signer/session split are not published. Static JPGs contain no
motion, while one example video per label cannot provide a signer-independent
training and validation split. Repeating or heavily augmenting those samples
would create leakage and misleading validation accuracy.

An honest retraining effort requires the original CADT dataset or a new
consented dataset with multiple independent signers and sessions for every
label. Until that exists, the released checkpoint is kept unchanged and
runtime uncertainty gates reject weak predictions.

## Accuracy strategy without retraining data

The browser keeps CADT's training-time contract unchanged: MediaPipe detection
order, wrist-relative XYZ values, 30 frames, and the released weights. It then
adds rejection outside the model: the winning class must clear both CADT's
confidence threshold and a top-two confidence margin, remain stable through a
temporal vote, and pass the hand-release gate before another word is accepted.
These gates target false activations; they cannot improve the checkpoint's
ability to distinguish two classes it learned poorly.

The next defensible model improvement is not I3D. It is to obtain the original
30x126 training clips (or collect replacements), add explicit idle/transition
clips, and evaluate by held-out signer. Only then should LSTM, GRU, TCN, and
RGB/landmark fusion alternatives be compared on the same split.
