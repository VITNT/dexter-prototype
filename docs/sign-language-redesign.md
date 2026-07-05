# Scalable Sign Language Recognition Redesign

This document proposes a realistic redesign for Dexter so it can move from
mostly static hand-pose detection to recorded-video sign recognition across
multiple sign languages.

## Confirmed Facts

### Confirmed From This Repository

- `script.js` is the production recognizer and is currently a single large
  browser module that owns UI state, camera control, MediaPipe setup,
  landmark rendering, feature extraction, routing, TensorFlow.js inference,
  voting, word building, and translation calls.
- ASL spelling is split between static `fingerpose` rules and a small sequence
  model. The checked-in `asl-labels.json` contains only `J`, `Z`, and `idle`,
  which means most ASL letters are treated as static hand shapes.
- Movement is represented as a fixed 30-frame window. That is useful, but the
  current runtime treats it as a sliding live-camera buffer rather than a
  first-class recorded sign clip.
- CSL support was previously inconsistent in naming. The UI says `CSL Words`,
  and the main app tries to load optional flat files such
  as `csl_labels.json` and `csl_model.json` before falling back to routed
  submodels in `csl-words/submodel-registry.json`.
- CSL word recognition currently depends on handshape/body-zone routing
  to many small submodels, such as `open_palm_face` and `thumb_up_chest`.
- `recorder.js` captures short keypoint sequences and downloads JSON files.
  `trainer.js` trains a TensorFlow.js LSTM in the browser from manually
  uploaded JSON files.
- `api/translate.js` is ASL-specific: it accepts an `aslString` and prompts a
  model to rewrite ASL gloss as English.

### Confirmed External Reference

- Paul Kelly's Sign Language AI site presents the relevant product pattern:
  the user records signing, the tool interprets it, and written results appear
  in the camera view. The public page confirms the UX pattern, not the private
  model architecture.
- Google MediaPipe Hand Landmarker and Pose Landmarker both support decoded
  video frames as input, not only live webcam streams. That makes recorded
  clip processing a good fit for the current browser stack.
- Public ASL datasets exist for video-based work. WLASL is a word-level ASL
  video dataset with 2,000 common ASL words. How2Sign is a continuous ASL
  dataset with more than 80 hours of videos and aligned modalities.
- Public CSL resources appear much less mature. A 2024 Khmer Sign Language
  Recognition System presentation describes data collection
  with Cambodian institutions, preprocessing, model training, and text/voice
  output, but this project should assume CSL data will need to be collected
  or curated carefully.

## Assumptions

- The next student-project version should solve isolated sign recognition
  first: one short recorded clip maps to one sign/gloss. Continuous sentence
  recognition can be added later.
- Browser-first TensorFlow.js is still acceptable for the prototype. A Python
  training pipeline can be added later if datasets grow beyond what browser
  training can handle.
- "CSL" is ambiguous without project context. In this repository it appears
  to be tied to Khmer/Cambodian assets, so the code should use explicit
  language IDs and display names so ASL and CSL do not blur together.

## Diagnosis: Why The Current Implementation Does Not Scale

The current application is a working prototype, but its core abstractions are
too narrow for real sign language recognition.

1. It is frame-first instead of clip-first.
   Static ASL spelling works by inspecting one hand pose at a time. Dynamic
   signs are only detected when a 30-frame live buffer crosses a movement
   threshold. Full signs should be modeled as complete time sequences with a
   known start, end, frame rate, and preprocessing contract.

2. Static and dynamic recognition are mixed together.
   `fingerpose`, movement thresholds, routed submodels, vote windows, and TFJS
   model calls all live together in `script.js`. Adding a new language or
   model means editing runtime logic rather than adding data/model metadata.

3. Language support is not data-driven.
   ASL and CSL are represented through hardcoded modes and filenames.
   Each language needs its own manifest that declares labels, model paths,
   landmark layout, translations, sample metadata, and output language.

4. The submodel routing approach will grow poorly.
   Routing by handshape and body zone can help small experiments, but it means
   every new bucket needs separate labels, model files, calibration, and
   fallback behavior. It also breaks when two signs share a route but differ
   by motion, orientation, facial expression, or two-hand timing.

5. The dataset is not a dataset yet.
   JSON downloads from the recorder contain labels and frames, but there is no
   durable catalog for sign language, signer, consent, sample path, raw video
   path, keypoint path, split, frame rate, translation, or label version.

6. Translation is ASL-only.
   `api/translate.js` assumes ASL gloss to English. A scalable system needs a
   translation layer that receives recognized glosses plus source language and
   target language, then chooses a direct dictionary translation, gloss
   rewrite, text-to-speech, or later a proper sign-language translation model.

7. Recorded video is not a first-class inference input.
   The main app is live camera only. Recorded clips should be captured or
   uploaded, sampled frame-by-frame, converted into keypoints, normalized,
   classified, and then translated.

## What To Keep, Refactor, Replace, Or Remove

### Keep

- MediaPipe landmark extraction, because it already gives useful hand and body
  features and works in the browser.
- TensorFlow.js inference for the first browser prototype.
- The idea of a 30-frame sequence and the existing LSTM trainer as a baseline.
- The dictionary/learning pages as a separate educational surface.

### Refactor

- Move MediaPipe setup and landmark extraction out of `script.js`.
- Move frame normalization, missing-landmark filling, resampling, and model
  input shaping into shared preprocessing modules.
- Replace hardcoded model selection with a model registry loaded from a catalog
  manifest.
- Split capture, inference, translation, and UI rendering into separate
  modules.

### Replace

- Replace the primary handshape-zone submodel strategy with language-level
  isolated-sign sequence models. Hierarchical routing can return later as an
  optimization, not as the core architecture.
- Replace direct JSON downloads as the only dataset format with a catalog that
  links raw video, extracted keypoints, labels, translations, and metadata.
- Replace ASL-only translation with a language-aware translation interface.

### Remove Later

- Remove static `fingerpose` as the main ASL alphabet classifier once enough
  recorded examples exist for all letters, including dynamic `J` and `Z`.
  Until then, it can remain as a low-risk fallback.
- Remove duplicate mode handlers (`selectMode` and `changeMode`) once the UI
  reads from the language/model registry.

## Proposed Architecture

The new system should treat every recognition request as a clip pipeline:

```text
Recorded video or live segment
  -> frame sampler
  -> MediaPipe hand/pose keypoint extractor
  -> sequence cleaner and normalizer
  -> fixed-length model input
  -> language-specific sign classifier
  -> gloss/sign result
  -> translation layer
  -> written text and optional speech output
```

### Core Modules

- `capture`: records a short clip with `MediaRecorder`, accepts uploaded video,
  or segments a live stream.
- `keypoints`: runs MediaPipe in `VIDEO` mode on decoded frames and emits a
  stable landmark layout.
- `preprocessing`: fills short missing-hand gaps, normalizes around body or
  wrist anchors, resamples to the model's target frame count, and validates
  feature width.
- `model-registry`: loads the correct model, labels, landmark layout, and
  target frame count for a selected language.
- `inference`: converts one preprocessed sequence into top-k predictions.
- `translation`: maps recognized labels/glosses to target written language and
  optional speech.
- `catalog`: stores languages, signs, samples, model metadata, translations,
  and dataset splits.

## Recommended ML Approach

### Version 1: Student-Project Baseline

Use MediaPipe hand plus pose keypoints and train an isolated-sign sequence
classifier.

- Input: 30 to 60 frames per sign at 10 to 15 FPS.
- Features: pose landmarks plus left/right hand landmarks. Start with the
  current 225-value layout: 33 pose points and 2 hands x 21 points, each with
  x/y/z.
- Preprocessing: missing-frame forward fill, coordinate normalization relative
  to upper body or wrist anchor, fixed-length resampling, optional mirroring
  and light spatial jitter.
- Model: LSTM, GRU, or Temporal CNN in TensorFlow.js.
- Output: one label/gloss plus confidence.

This is realistic because the current trainer already uses LSTM sequence input.
The main change is making sequence data and language manifests first-class.

### Model Tradeoffs

- LSTM: strong baseline for short ordered keypoint sequences, easy to explain,
  already close to the current trainer.
- GRU: similar to LSTM with fewer parameters, useful when the dataset is small.
- Temporal CNN: often faster than recurrent models and good for fixed-length
  motion patterns.
- Transformer: useful for larger datasets and variable-length clips, but easier
  to overfit in a small student dataset.
- 3D CNN: learns directly from RGB video, but needs much more data and compute.
  Keep it as a later experiment, not the first architecture.
- Pose graph or spatio-temporal graph models: promising for body/hand
  trajectories, but probably too much complexity for the first refactor.

## Data Structure

Use a catalog manifest as the source of truth. See
`data/catalog/sign-catalog.example.json` for a concrete starter schema.

Each sign entry should store:

- language ID, such as `asl` or `csl`
- stable sign ID, such as `asl:j` or `csl:thank_you`
- human label/gloss
- sign type: `static`, `dynamic`, or `compound`
- translations by target language
- raw video sample paths
- extracted keypoint paths
- signer/session metadata
- split: `train`, `validation`, or `test`
- model compatibility metadata: landmark layout, frame count, FPS

## Movement-Based Sign Handling

Movement signs should be recognized from sequences, not final hand positions.

- Capture a full clip, usually 2 to 10 seconds.
- Sample frames at a fixed rate.
- Extract hands and pose for every sampled frame.
- Convert variable-length clips into a model length with resampling, not by
  dropping everything after 30 live frames.
- Keep temporal signals such as velocity, direction, and relative hand motion
  inside the frame sequence. Avoid hand-coded movement thresholds except for
  idle detection or segmentation.
- Evaluate complete clips during validation so accuracy reflects actual sign
  recognition, not repeated predictions from overlapping live windows.

## Recorded Video Flow

1. User chooses language and target output language.
2. User records a short clip or uploads a video.
3. The browser decodes the clip and samples frames.
4. MediaPipe extracts hands and upper-body pose from each frame.
5. The preprocessing module validates feature width, fills short gaps,
   normalizes coordinates, and resamples to the model's expected length.
6. The model registry loads the language-specific sequence model.
7. The inference module returns top-k labels and confidence.
8. The translation module turns the label/gloss into written text.
9. Optional: browser speech synthesis reads the translated text.

## Suggested Folder Structure

```text
.
├── api/
│   └── translate.js
├── data/
│   ├── catalog/
│   │   └── sign-catalog.example.json
│   ├── raw/
│   │   └── <language>/<label>/<sample>.webm
│   └── keypoints/
│       └── <language>/<label>/<sample>.json
├── docs/
│   └── sign-language-redesign.md
├── models/
│   └── <language>/<model-version>/
│       ├── model.json
│       ├── weights.bin
│       └── labels.json
├── src/
│   ├── capture/
│   ├── keypoints/
│   ├── pipeline/
│   ├── translation/
│   └── ui/
├── index.html
├── recorder.html
├── trainer.html
└── styles.css
```

The files added in this branch start with `src/pipeline/` because that is the
highest-value boundary to extract first.

## Migration Plan

### Do Now

- Create the new branch before changes.
- Document the architecture and current scalability limits.
- Introduce a catalog schema for languages, signs, samples, keypoints, models,
  and translations.
- Introduce reusable preprocessing and model-registry starter modules.
- Treat ASL `J` and `Z`, CSL words, and future languages as sequence
  classification problems.

### Do Next

- Update the recorder to save raw video and keypoints together, not keypoints
  only.
- Add signer/session metadata and train/validation/test split fields.
- Update the trainer to read the catalog manifest and train one model per
  language/vocabulary.
- Add a recorded-video UI path to `index.html`: record/upload, process clip,
  show top-k results, then translate.
- Make `api/translate.js` accept `{ sourceLanguage, targetLanguage, glosses }`.

### Do Later

- Add continuous sign segmentation for multi-sign sentences.
- Add face landmarks or expression features where linguistically needed.
- Move larger training runs to Python if browser training becomes slow or
  memory-limited.
- Experiment with Temporal CNN, Transformer, or graph-based models once there
  are enough samples.
- Add model evaluation reports with signer-independent splits.

### Requires More Data

- CSL recognition needs collaboration with fluent signers,
  teachers, or local institutions. The model cannot become reliable from only
  a few one-person samples.
- Each sign should include different signers, lighting, camera angles,
  clothing, speeds, and dialect/variant notes.
- Translation quality requires language expertise. A gloss label is not always
  equivalent to a natural spoken-language sentence.

## Risks And Limitations

- Sign languages are independent languages, not visual encodings of spoken
  languages. ASL and CSL need separate datasets and label systems.
- Some signs differ by facial expression, mouth movement, body posture, or
  context. Hands-only data will not be enough for all signs.
- Small datasets can show high validation accuracy while failing on new users.
  Use signer-independent validation as soon as possible.
- Label ambiguity is real. Two signs may share an English gloss, and one sign
  may translate differently depending on sentence context.
- Public ASL datasets do not solve CSL coverage. They can help with
  architecture and pretraining ideas, but local data is still required.
- The first version should be presented as an assistive learning prototype, not
  as a replacement for human interpreters.

## Starter Code Added In This Branch

- `src/pipeline/sequence_preprocessing.mjs` prepares variable-length keypoint
  sequences for model input.
- `src/pipeline/model_registry.mjs` resolves a language model from a catalog.
- `src/pipeline/recorded_video_pipeline.mjs` sketches the recorded-video
  inference flow.
- `data/catalog/sign-catalog.example.json` shows how signs, samples,
  translations, and model metadata can be stored.

## Source Notes

- Sign Language AI by Paul Kelly: https://www.signlanguageai.com/
- MediaPipe Hand Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- WLASL: https://dxli94.github.io/WLASL/
- How2Sign: https://how2sign.github.io/
- Khmer Sign Language Recognition System presentation:
  https://naivo.org/index.php/2024forum/presentations/download/146/presentation
