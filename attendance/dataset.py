from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from attendance.identifiers import validate_student_id

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


def _session_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _sessions_for_person(person_dir: Path) -> list[tuple[str, list[Path]]]:
    sessions = []
    for session_dir in sorted(path for path in person_dir.iterdir() if path.is_dir()):
        images = _session_images(session_dir)
        if images:
            sessions.append((session_dir.name, images))
    if not sessions:
        raise ValueError(
            f"No capture-session folders found under {person_dir}. "
            "Use one subfolder per recording/session to prevent leakage."
        )
    return sessions


def _split_counts(session_count: int) -> tuple[int, int, int]:
    if session_count < 3:
        raise ValueError(
            "Each enrolled student needs at least three independent capture-session "
            "folders (train, validation, test)."
        )
    train_count = max(1, int(session_count * 0.6))
    validation_count = max(1, int(session_count * 0.2))
    if train_count + validation_count >= session_count:
        train_count = session_count - 2
    return train_count, validation_count, session_count - train_count - validation_count


def prepare_splits(
    students_dir: str | Path,
    unknown_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    split_data = {split: [] for split in SPLITS}
    person_dirs = sorted(path for path in Path(students_dir).iterdir() if path.is_dir())
    if not person_dirs:
        raise ValueError(f"No student ID folders found in {students_dir}.")

    for person_dir in person_dirs:
        validate_student_id(person_dir.name)
        sessions = _sessions_for_person(person_dir)
        counts = _split_counts(len(sessions))
        rng.shuffle(sessions)
        offset = 0
        for split, count in zip(SPLITS, counts):
            for session_id, images in sessions[offset : offset + count]:
                split_data[split].extend(
                    {"student_id": person_dir.name, "path": str(path.resolve())}
                    for path in images
                )
            offset += count

    unknown_root = Path(unknown_dir)
    if unknown_root.is_dir():
        for person_dir in sorted(path for path in unknown_root.iterdir() if path.is_dir()):
            sessions = _sessions_for_person(person_dir)
            if len(sessions) < 2:
                raise ValueError(
                    f"Unknown identity {person_dir.name!r} needs at least two "
                    "independent sessions for validation and test."
                )
            rng.shuffle(sessions)
            for split, session_index in (("validation", 0), ("test", 1)):
                _, images = sessions[session_index]
                split_data[split].extend(
                    {"student_id": "Unknown", "path": str(path.resolve())}
                    for path in images
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    logger.info("Session-disjoint data split written to %s", output_path)
    return split_data


def load_splits(path: str | Path) -> dict:
    split_data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(split_data, dict) or any(
        split not in split_data for split in SPLITS
    ):
        raise ValueError(f"Invalid split manifest: {path}")
    paths_by_split = {
        split: {record["path"] for record in split_data[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = paths_by_split[left] & paths_by_split[right]
            if overlap:
                raise ValueError(f"Image leakage between {left} and {right} splits.")
    return split_data
