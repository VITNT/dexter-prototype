from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CONTINUOUS_CHECKPOINT_PATH
from .continuous_dataset import ContinuousAslDataset
from .continuous_model import build_wlasl_ctc_model
from .inference import load_labels


def train(
    manifest: Path,
    output: Path,
    epochs: int,
    learning_rate: float,
    unfreeze_encoder: bool,
    resume: bool,
    device_name: str,
) -> None:
    vocabulary = load_labels()
    dataset = ContinuousAslDataset(manifest, vocabulary)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = build_wlasl_ctc_model()
    if resume:
        if not output.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint does not exist: {output}")
        checkpoint = torch.load(output, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
    device = select_device(device_name)
    print(f"device={device}")
    model.to(device)

    if not unfreeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_function = nn.CTCLoss(blank=model.blank_index, zero_infinity=True)

    for epoch in range(epochs):
        model.train()
        if not unfreeze_encoder:
            model.encoder.eval()
        running_loss = 0.0
        for video, targets in loader:
            # batch_size=1 keeps variable-length sentence videos simple and
            # avoids treating padded RGB frames as real temporal evidence.
            video = video[0].unsqueeze(0).to(device)
            targets = targets[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(video)
            input_lengths = torch.tensor([log_probabilities.shape[0]], dtype=torch.long)
            target_lengths = torch.tensor([targets.numel()], dtype=torch.long)
            loss = loss_function(
                log_probabilities,
                targets,
                input_lengths,
                target_lengths,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach())
        print(f"epoch={epoch + 1} loss={running_loss / len(loader):.5f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "dexter-wlasl-ctc-v1",
        "vocabulary": vocabulary,
        "blank_index": model.blank_index,
        "model_state": model.state_dict(),
        "training_manifest": str(manifest.resolve()),
        "synthetic_only": all(
            '"source": "synthetic_wlasl_concatenation"' in line
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
    }, output)
    print(f"saved={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WLASL-pretrained ASL CTC recognizer.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=CONTINUOUS_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Training device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    args = parser.parse_args()
    train(
        args.manifest.resolve(),
        args.output.resolve(),
        args.epochs,
        args.learning_rate,
        args.unfreeze_encoder,
        args.resume,
        args.device,
    )


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS was requested but is unavailable.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
