from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .continuous_preprocessing import prepare_continuous_video


class ContinuousAslDataset(Dataset):
    """
    JSONL entries may reference a real sentence (`video`) or a synthetic WLASL
    sequence (`videos`). Both require an ordered `gloss` array.
    """

    def __init__(self, manifest_path: str | Path, vocabulary: list[str]):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.entries = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.label_to_index = {label: index for index, label in enumerate(vocabulary)}
        if not self.entries:
            raise ValueError("The continuous ASL manifest is empty.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[index]
        glosses = entry.get("gloss")
        if not isinstance(glosses, list) or not glosses:
            raise ValueError(f"Manifest entry {index} needs a non-empty gloss array.")
        try:
            targets = torch.tensor([self.label_to_index[label] for label in glosses], dtype=torch.long)
        except KeyError as error:
            raise ValueError(f"Unknown gloss in manifest entry {index}: {error.args[0]}") from error

        paths = entry.get("videos") or [entry.get("video")]
        if not paths or any(not path for path in paths):
            raise ValueError(f"Manifest entry {index} needs video or videos.")
        max_seconds = 3.0 if entry.get("source") == "synthetic_wlasl_concatenation" else 10.0
        clips = [
            prepare_continuous_video(str(self._resolve(path)), max_seconds=max_seconds)[0][0]
            for path in paths
        ]
        if len(clips) == 1:
            video = clips[0]
        else:
            gap = torch.zeros((clips[0].shape[0], 4, clips[0].shape[2], clips[0].shape[3]))
            pieces: list[torch.Tensor] = []
            for clip_index, clip in enumerate(clips):
                if clip_index:
                    pieces.append(gap)
                pieces.append(clip)
            video = torch.cat(pieces, dim=1)
        return video, targets

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate
