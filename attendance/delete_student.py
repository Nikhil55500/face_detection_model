from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from attendance.config import load_config
from attendance.identifiers import validate_student_id
from attendance.storage import (
    invalidate_evaluation_reports,
    load_gallery,
    save_gallery,
)

logger = logging.getLogger(__name__)


def delete_student(config_path: str, student_id: str, delete_images: bool) -> None:
    validate_student_id(student_id)
    config, _ = load_config(config_path)
    gallery_path = Path(config["paths"]["templates"])
    students_root = Path(config["paths"]["students"]).resolve()
    image_directory = students_root / student_id
    if delete_images and image_directory.exists():
        if image_directory.is_symlink() or image_directory.resolve().parent != students_root:
            raise ValueError(f"Refusing to delete images outside the student data root: {image_directory}")
    gallery = load_gallery(gallery_path)
    if student_id not in gallery:
        raise ValueError(f"Student {student_id!r} is not enrolled.")
    del gallery[student_id]
    save_gallery(gallery_path, gallery)
    if delete_images:
        if image_directory.exists():
            shutil.rmtree(image_directory)
            logger.info("Deleted source images for %s from %s", student_id, image_directory)
    split_manifest = Path(config["paths"]["split_manifest"])
    if split_manifest.is_file():
        split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
        for split, records in split_data.items():
            split_data[split] = [
                record for record in records if record["student_id"] != student_id
            ]
        split_manifest.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    scene_manifest_value = config["paths"].get("classroom_scenes")
    if scene_manifest_value and Path(scene_manifest_value).is_file():
        scene_manifest = Path(scene_manifest_value)
        scene_data = json.loads(scene_manifest.read_text(encoding="utf-8"))
        for split, scenes in scene_data.items():
            for scene in scenes:
                scene["faces"] = [
                    face for face in scene["faces"] if face["student_id"] != student_id
                ]
        scene_manifest.write_text(json.dumps(scene_data, indent=2), encoding="utf-8")
    invalidate_evaluation_reports(config["paths"]["reports"])
    logger.info("Deleted encrypted templates for %s.", student_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete one student's encrypted templates.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--delete-images", action="store_true",
                        help="Also permanently delete data/students/<student-id> source images.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    arguments = parser.parse_args()
    if not arguments.yes:
        action = "templates and source images" if arguments.delete_images else "templates"
        confirmation = input(
            f"Delete {action} for student {arguments.student_id!r}? Type DELETE to confirm: "
        )
        if confirmation != "DELETE":
            raise SystemExit("Deletion cancelled.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    delete_student(arguments.config, arguments.student_id, arguments.delete_images)


if __name__ == "__main__":
    main()
