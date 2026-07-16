# OpenHands ST-GCN WLASL2000 integration

## Runtime contract

ASL Words uses the released pose-based AI4Bharat/OpenHands ST-GCN checkpoint.

| Stage | Contract |
| --- | --- |
| Browser tracking | MediaPipe Pose (33) + left hand (21) + right hand (21) |
| Capture window | Continuous 2.5 seconds |
| Landmark selection | OpenHands `mediapipe_holistic_minimal_27` preset |
| Spatial preprocessing | Clip-level shoulder midpoint and mean shoulder-width normalization |
| Model input | `[batch, 2, variable frames, 27]` float32 |
| Model output | 2,000 logits in alphabetically sorted WLASL gloss order |
| API | multipart `landmarks` JSON field on `POST /predict` |

The 27 selected points are MediaPipe holistic indices
`[0,2,5,11,12,13,14,33,37,38,41,42,45,46,49,50,53,54,58,59,62,63,66,67,70,71,74]`.
Only x/y coordinates enter the model, matching the released checkpoint.

## Checkpoint contract

The service strictly loads
`asl_service/models/openhands_wlasl2000_stgcn.ckpt` into the OpenHands topology:

- ten ST-GCN blocks with learned edge importance
- 27 graph nodes and three spatial adjacency partitions
- 2 input channels
- 256-dimensional pooled encoder output
- fully connected 2,000-class decoder with dropout `0.05`

The supplied archive checksum is
`7a86185715e1a63a16dd87fbf6433e86bf1be4627193c181b148398a8603bc7a`.
The loader uses an allowlisted PyTorch `weights_only=True` path for the two
Lightning callback records, avoiding unrestricted checkpoint unpickling.

## Acceptance checks

- Confirm `/health` reports `model: "OpenHands ST-GCN"`, 2,000 classes, and
  model input shape `[2, "frames", 27]`.
- Confirm the checkpoint passes strict key and tensor-shape validation.
- Re-evaluate motion, confidence, and top-two-margin gates on held-out webcam
  recordings because WLASL has no idle/no-sign class.
- Keep the WLASL usage terms in scope when distributing or deploying the model.
