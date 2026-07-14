from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def load_wlasl_clips(
    dataset_root: Path,
    metadata_path: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, list[Path]]:
    if metadata_path is None:
        return {
            directory.name: sorted(
                path for path in directory.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES
            )
            for directory in dataset_root.iterdir()
            if directory.is_dir()
        }

    allowed_labels = None
    if labels_path:
        allowed_labels = {
            line.split(maxsplit=1)[1]
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(line.split(maxsplit=1)[0]) < 300
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clips_by_gloss: dict[str, list[Path]] = {}
    for entry in metadata:
        gloss = entry["gloss"]
        if allowed_labels is not None and gloss not in allowed_labels:
            continue
        for instance in entry.get("instances", []):
            video_id = instance.get("video_id")
            if not video_id:
                continue
            for suffix in VIDEO_SUFFIXES:
                candidate = dataset_root / f"{video_id}{suffix}"
                if candidate.exists():
                    clips_by_gloss.setdefault(gloss, []).append(candidate)
                    break
    return clips_by_gloss


def build_manifest(
    dataset_root: Path,
    output_path: Path,
    count: int,
    seed: int,
    metadata_path: Path | None = None,
    labels_path: Path | None = None,
) -> None:
    clips_by_gloss = load_wlasl_clips(dataset_root, metadata_path, labels_path)
    clips_by_gloss = {label: paths for label, paths in clips_by_gloss.items() if paths}
    if len(clips_by_gloss) < 2:
        raise ValueError("Expected WLASL clips arranged as <root>/<gloss>/<video>.")

    randomizer = random.Random(seed)
    labels = sorted(clips_by_gloss)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for index in range(count):
            word_count = randomizer.randint(2, min(5, len(labels)))
            gloss = randomizer.sample(labels, word_count)
            videos = [str(randomizer.choice(clips_by_gloss[label]).resolve()) for label in gloss]
            output.write(json.dumps({
                "id": f"synthetic-wlasl-{index:06d}",
                "source": "synthetic_wlasl_concatenation",
                "videos": videos,
                "gloss": gloss,
            }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synthetic multiword CTC examples from isolated WLASL clips."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="WLASL_v0.3.json; enables the official flat videos/<video_id>.mp4 layout.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt"),
        help="Official class list used to keep only WLASL300 glosses.",
    )
    args = parser.parse_args()
    build_manifest(
        args.dataset_root.resolve(),
        args.output.resolve(),
        args.count,
        args.seed,
        args.metadata.resolve() if args.metadata else None,
        args.labels.resolve() if args.labels else None,
    )


if __name__ == "__main__":
    main()
