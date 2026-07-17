# TGCN WLASL2000 integration

## Runtime contract

ASL Words uses the pretrained `sharonn18/tgcn-wlasl` ASL2000 checkpoint from
Hugging Face. It is the pose-based TGCN architecture from the original WLASL
repository.

| Stage | Contract |
| --- | --- |
| Browser tracking | MediaPipe Pose (33) + left hand (21) + right hand (21) |
| Capture window | Continuous 2.5 seconds |
| Landmark mapping | OpenPose BODY_25 retained 13 points + both 21-point hands |
| Spatial preprocessing | MediaPipe normalized x/y converted from `[0,1]` to `[-1,1]` |
| Temporal preprocessing | One padded view or four fixed 50-frame test views |
| Model input | `[views, 55, 100]` float32 |
| Model output | 2,000 logits in alphabetically sorted WLASL gloss order |
| API | multipart `landmarks` JSON field on `POST /predict` |

The 13 body/face nodes are nose, neck, right shoulder/elbow/wrist, left
shoulder/elbow/wrist, mid-hip, right eye, left eye, right ear, and left ear.
Neck and mid-hip are synthesized as MediaPipe shoulder and hip midpoints. The
42 hand points retain MediaPipe's left-hand-then-right-hand ordering, matching
OpenPose's per-hand landmark order.

For clips longer than 50 frames, the service uses the original WLASL
`k_copies_fixed_length_sequential_sampling` strategy and averages the four
view logits before softmax. Short clips repeat the last frame to reach 50.

## Checkpoint contract

The service strictly loads `asl_service/models/tgcn_wlasl2000.bin` into:

- a learned 55×55 attention matrix per graph-convolution layer
- a 100-to-256 input graph convolution
- 24 two-layer residual graph-convolution blocks
- mean pooling over 55 nodes
- a 256-to-2,000 output classifier

The pinned Hugging Face revision and checkpoint checksum are recorded in
`asl_service/models/README.md`. The loader uses `weights_only=True`; it never
enables unrestricted checkpoint unpickling.

## Acceptance checks

- Confirm `/health` reports `model: "TGCN WLASL2000"`, 2,000 classes, and
  model input shape `[55, 100]`.
- Confirm the checkpoint passes strict key and tensor-shape validation.
- Re-evaluate motion, confidence, and top-two-margin gates on held-out webcam
  recordings because WLASL has no idle/no-sign class.
- Keep WLASL dataset access and usage terms in scope when distributing or
  deploying the model.
