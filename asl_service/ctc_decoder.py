from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DecodedToken:
    label: str
    confidence: float
    start_step: int
    end_step: int


def greedy_ctc_decode(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    blank_index: int | None = None,
) -> list[DecodedToken]:
    """Collapse repeated frame predictions and CTC blanks into ordered tokens."""
    if blank_index is None:
        blank_index = len(labels)
    decoded: list[DecodedToken] = []
    active_index: int | None = None
    active_scores: list[float] = []
    active_start = 0

    def flush(end_step: int) -> None:
        nonlocal active_index, active_scores
        if active_index is not None and active_index != blank_index and active_scores:
            decoded.append(DecodedToken(
                label=labels[active_index],
                confidence=sum(active_scores) / len(active_scores),
                start_step=active_start,
                end_step=end_step,
            ))
        active_index = None
        active_scores = []

    for step, row in enumerate(probabilities):
        if not row:
            continue
        index = max(range(len(row)), key=row.__getitem__)
        score = float(row[index])
        if index == active_index:
            active_scores.append(score)
            continue
        flush(step)
        if index != blank_index:
            active_index = index
            active_scores = [score]
            active_start = step
    flush(len(probabilities))
    return decoded


def tokens_with_timestamps(
    tokens: Sequence[DecodedToken],
    total_steps: int,
    duration_ms: int,
) -> list[dict]:
    if total_steps < 1:
        return []
    return [{
        "word": token.label,
        "confidence": token.confidence,
        "startMs": round(token.start_step * duration_ms / total_steps),
        "endMs": round(token.end_step * duration_ms / total_steps),
    } for token in tokens]
