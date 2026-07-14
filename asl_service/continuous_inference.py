from __future__ import annotations

import torch

from .continuous_model import load_continuous_model
from .continuous_preprocessing import prepare_continuous_video
from .ctc_decoder import greedy_ctc_decode, tokens_with_timestamps


def predict_continuous_video(video_path: str) -> dict:
    model, vocabulary, device = load_continuous_model()
    clip, duration_ms = prepare_continuous_video(video_path)
    with torch.inference_mode():
        log_probabilities = model(clip.to(device))
        probabilities = log_probabilities.exp()[:, 0, :].cpu().tolist()
    decoded = greedy_ctc_decode(probabilities, vocabulary, model.blank_index)
    words = tokens_with_timestamps(decoded, len(probabilities), duration_ms)
    return {
        "gloss": " ".join(word["word"] for word in words),
        "words": words,
        "durationMs": duration_ms,
        "decoder": "greedy_ctc_v1",
        "trainingDomain": model.training_domain,
    }
