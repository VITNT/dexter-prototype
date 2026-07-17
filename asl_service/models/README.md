# Hugging Face TGCN WLASL2000 checkpoint

The ASL-word service uses the `asl2000` checkpoint from:

<https://huggingface.co/sharonn18/tgcn-wlasl>

The pinned artifact is stored as `asl_service/models/tgcn_wlasl2000.bin` with
Git LFS. Its verified contract is:

- Hugging Face revision: `dacb4568719caa03c44764034f599a9f8a0f63f4`
- upstream file: `checkpoints/asl2000/pytorch_model.bin`
- file size: `26,491,626` bytes
- SHA-256: `23a994294be6156bcebb4d333a891dc8594a83f461b58b84068472729f5ba929`
- output classes: `2,000`
- input: `[batch, 55, 100]` (`50` frames with interleaved x/y coordinates)
- hidden size: `256`
- residual graph-convolution stages: `24`
- dropout: `0.3`

Run `git lfs pull` after cloning. A valid working-tree checkpoint is about 25
MiB, not a small text pointer. The service loads it with PyTorch's restricted
`weights_only=True` path and validates every parameter key and shape strictly.
