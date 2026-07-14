# Continuous ASL and segmented KSL

Dexter uses different sentence strategies because the available training data
has different supervision.

## Runtime behavior

- **KSL Words** records up to 10 seconds, detects a word whenever either hand is
  present, and closes the word after 400 ms with both hands lowered. Each range
  is padded by 120 ms and passed independently to the existing KSL classifier.
- **ASL Words** sends the complete clip to `/predict-sequence` when
  `asl_service/models/wlasl300_ctc.pt` is installed. The model emits an ordered
  gloss sequence with CTC timestamps. Until that checkpoint exists, the service
  falls back to the original one-word WLASL endpoint.

## Why WLASL is pretraining, not sentence supervision

WLASL contains isolated word clips. Those clips teach the I3D encoder useful
visual sign features, but they do not contain natural word transitions or an
alignment between a sentence video and several glosses. The included bootstrap
therefore concatenates WLASL clips into synthetic sequences and trains the CTC
head. Treat that checkpoint as a pipeline test, not a production sentence
model. Fine-tune it on real continuous ASL recordings before evaluating natural
signing.

## 1. Build a synthetic WLASL bootstrap manifest

The official downloader produces processed clips under
`third_party/WLASL/start_kit/videos/<video_id>.mp4`. Build the synthetic manifest
directly from that flat directory and the official metadata:

```bash
.venv/bin/python -m asl_service.build_synthetic_wlasl_manifest \
  third_party/WLASL/start_kit/videos data/asl-synthetic.jsonl \
  --metadata third_party/WLASL/start_kit/WLASL_v0.3.json --count 5000
```

The builder keeps only the 300 labels supported by `wlasl300.pt`. It also still
accepts a manually arranged `<root>/<gloss>/<video>` directory when `--metadata`
is omitted.

## 2. Train the frozen-encoder CTC bootstrap

Install the official WLASL source and `wlasl300.pt` as described in
`asl_service/models/README.md`, then run:

```bash
.venv/bin/python -m asl_service.train_continuous \
  data/asl-synthetic.jsonl --epochs 20
```

`--device auto` is the default and uses Apple MPS on a compatible Mac. Pass
`--device cpu` if the local PyTorch build reports an unsupported MPS operation.

This writes `asl_service/models/wlasl300_ctc.pt`. Restart the FastAPI service;
`/health` should return `"sequenceModelReady": true`.

## 3. Prepare real continuous clips

Prepare a JSONL manifest from an external, reviewed continuous-ASL dataset. The
contract is one object per line:

```json
{"video":"sentence.webm","gloss":["ME","WANT","GO"]}
```

## 4. Fine-tune on real continuous ASL

Start from a low learning rate and unfreeze the WLASL encoder:

```bash
python -m asl_service.train_continuous data/asl-continuous.jsonl \
  --epochs 30 --learning-rate 0.0001 --unfreeze-encoder --resume
```

For a production experiment, add signer-independent train/dev splits, compute
word error rate, and use beam-search decoding with a gloss language model.

## KSL migration path

Continue using segmented KSL inference until a separately collected continuous
KSL dataset is large enough for signer-independent CTC evaluation to outperform
the isolated-word baseline.
