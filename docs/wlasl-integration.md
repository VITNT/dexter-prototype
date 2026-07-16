# PhonSSM WLASL300 integration

## Runtime contract

ASL Words now uses a local PyTorch PhonSSM service rather than the pretrained
RGB I3D model.

| Stage | Contract |
| --- | --- |
| Browser tracking | MediaPipe Pose (33) + left hand (21) + right hand (21) |
| Capture window | Continuous 2.5 seconds, gated by sustained hand visibility |
| Temporal preprocessing | Uniform sample or edge-pad to 30 frames |
| Spatial preprocessing | Shoulder midpoint origin, shoulder-width scale |
| Model input | `[batch, 30, 225]` float32 |
| Model output | 300 logits in WLASL v0.3 top-300 order |
| API | multipart `landmarks` JSON field on `POST /predict` |

The browser uses stable semantic left/right slots and never copies one detected
hand into both slots. The backend preprocessing mirrors the reference
`training/preprocess_wlasl_full.py` implementation.

## Checkpoint contract

The service loads `asl_service/models/phonssm_wlasl300.pt` strictly as the
reference pose+hands configuration:

- `input_mode="pose_hands"`
- `num_landmarks=75`
- `num_frames=30`
- `num_signs=300`
- `temperature=0.5`

The repository includes the authoritative 300-label order but not a model
weight. The linked PhonSSM repository reports 74.41% WLASL300 top-1 accuracy,
but its benchmark `best_model.pt` is ignored and absent from Git. Do not use its
tracked 5,565-class dominant-hand checkpoint; its input and output contracts are
different.

## Acceptance checks

- Confirm `/health` reports `model: "PhonSSM"`, 300 classes, and input shape
  `[30, 225]`.
- Load the checkpoint with strict key/shape validation.
- Compare logits against the reference repository on the same preprocessed
  landmark sequence before trusting production results.
- Re-evaluate the motion, top-1 confidence, and top-two margin gates on held-out
  webcam recordings; WLASL has no idle class.
- Test signers absent from training and keep the WLASL usage agreement in scope
  if the benchmark data is used to reproduce the checkpoint.
