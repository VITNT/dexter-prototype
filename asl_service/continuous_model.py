from __future__ import annotations

import sys
from functools import lru_cache

import torch
from torch import nn

from .config import (
    CHECKPOINT_PATH,
    CONTINUOUS_CHECKPOINT_PATH,
    NUM_CLASSES,
    OFFICIAL_I3D_ROOT,
)
from .inference import load_labels


class WlaslCtcRecognizer(nn.Module):
    """WLASL-pretrained RGB encoder with a bidirectional temporal CTC head."""

    def __init__(self, encoder: nn.Module, vocabulary_size: int, hidden_size: int = 256):
        super().__init__()
        self.encoder = encoder
        self.temporal = nn.GRU(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocabulary_size + 1)
        self.blank_index = vocabulary_size

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.encoder.extract_features(video)
        features = features.mean(dim=(-1, -2)).transpose(1, 2)
        sequence, _ = self.temporal(features)
        return self.classifier(sequence).log_softmax(dim=-1).transpose(0, 1)


def load_wlasl_encoder() -> nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"WLASL300 checkpoint not found at {CHECKPOINT_PATH}.")
    if not OFFICIAL_I3D_ROOT.exists():
        raise FileNotFoundError("The official WLASL I3D source is not installed.")
    sys.path.insert(0, str(OFFICIAL_I3D_ROOT))
    from pytorch_i3d import InceptionI3d

    encoder = InceptionI3d(400, in_channels=3)
    encoder.replace_logits(NUM_CLASSES)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    encoder.load_state_dict(state)
    return encoder


def build_wlasl_ctc_model() -> WlaslCtcRecognizer:
    return WlaslCtcRecognizer(load_wlasl_encoder(), len(load_labels()))


@lru_cache(maxsize=1)
def load_continuous_model():
    if not CONTINUOUS_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Continuous ASL checkpoint not found at {CONTINUOUS_CHECKPOINT_PATH}. "
            "Run asl_service/train_continuous.py first."
        )
    checkpoint = torch.load(CONTINUOUS_CHECKPOINT_PATH, map_location="cpu")
    vocabulary = checkpoint.get("vocabulary") or load_labels()
    model = WlaslCtcRecognizer(load_wlasl_encoder(), len(vocabulary))
    model.load_state_dict(checkpoint["model_state"])
    model.training_domain = (
        "synthetic_wlasl_only" if checkpoint.get("synthetic_only") else "continuous_asl_finetuned"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, vocabulary, device
