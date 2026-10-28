from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import cv2

from attendance.config import load_config
from attendance.dataset import load_splits
from attendance.face_engine import FaceEngine
from attendance.identifiers import validate_student_id
from attendance.storage import (
    invalidate_evaluation_reports,
    load_gallery,
    save_gallery,
)

logger = logging.getLogger(__name__)


def enroll(config_path: str, student_id: str | None = None) -> None:
    config, _ = load_config(config_path)
    if student_id:
        validate_student_id(student_id)
    split_data = load_splits(config["paths"]["split_manifest"])
    training_records = [
        record
        for record in split_data["train"]
        if student_id is None or record["student_id"] == student_id
    ]
    if not training_records:
        raise ValueError("No training images found. Prepare the dataset splits first.")

    engine = FaceEngine(config)
    templates: dict[str, list] = defaultdict(list)
    rejected = 0
    for record in training_records:
        image_path = Path(record["path"])
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Skipping unreadable image: %s", image_path)
            rejected += 1
            continue
        observations = engine.extract(image)
        if len(observations) != 1:
            logger.warning("Expected one face, found %d: %s", len(observations), image_path)
            rejected += 1
            continue
        observation = observations[0]
        if observation.review_reason:
            logger.warning("Rejected low-quality enrollment image (%s): %s",
                           observation.review_reason, image_path)
            rejected += 1
            continue
        templates[record["student_id"]].append(observation.embedding)

    expected_ids = {record["student_id"] for record in training_records}
    for enrolled_id in expected_ids:
        validate_student_id(enrolled_id)
    missing_ids = expected_ids - templates.keys()
    if missing_ids:
        raise ValueError(
            "No usable enrollment image for: "
            + ", ".join(sorted(missing_ids))
            + ". Improve image quality and try again."
        )

    gallery = load_gallery(config["paths"]["templates"])
    gallery.update(templates)
    save_gallery(config["paths"]["templates"], gallery)
    invalidate_evaluation_reports(config["paths"]["reports"])
    logger.info(
        "Enrolled %d student(s), saved %d embeddings; rejected %d images.",
        len(templates),
        sum(map(len, templates.values())),
        rejected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate encrypted face templates from training-split images only."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--student-id", help="Only enroll or refresh this student.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    enroll(arguments.config, arguments.student_id)


if __name__ == "__main__":
    main()
