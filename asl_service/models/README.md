# Install the PhonSSM WLASL300 checkpoint

The service expects this exact file:

`asl_service/models/phonssm_wlasl300.pt`

It must be the pose+hands WLASL300 checkpoint produced by the reference
`training/benchmark_external.py --dataset wlasl --subset 300` run, with:

- 300 output classes
- input mode `pose_hands`
- 75 landmarks × 3 coordinates
- 30 frames
- classifier temperature `0.5`

The vendored architecture is pinned to upstream commit
`5169623e57939d08f71eb08aad94f448c89c737d`; checkpoints from a different
revision must still pass strict state-dictionary validation.

The loader accepts the reference checkpoint wrapper containing
`model_state_dict`, a `state_dict` wrapper, or a raw PyTorch state dictionary.
It validates every key and tensor shape strictly before the service starts
classifying.

## Availability limitation

<https://github.com/anonymoussubmitter-167/phon-ssm> includes the WLASL300
metrics and training history but not the corresponding weight. Its benchmark
checkpoint path is excluded by the repository's `*.pt` ignore rule. The only
tracked PhonSSM checkpoint is for the separate 5,565-class, dominant-hand model
and is not compatible with this service.

Obtain the WLASL300 `best_model.pt` from the authors or reproduce the benchmark
training run, then copy it here and rename it to `phonssm_wlasl300.pt`.

The old RGB I3D file `wlasl300.pt` is no longer loaded. Model weights remain
ignored by Git; do not commit them without confirming redistribution rights.
