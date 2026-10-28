from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from cryptography.fernet import Fernet

from attendance.dataset import load_splits, prepare_splits
from attendance.delete_student import delete_student
from attendance.evaluate import (
    _extract_scene_records,
    _match_scene_faces,
    _scores_for_threshold,
)
from attendance.face_engine import FaceObservation
from attendance.identifiers import validate_student_id
from attendance.recognize import IdentityTracker, classify
from attendance.storage import (
    load_gallery,
    normalized_prototypes,
    rank_matches,
    save_gallery,
)


class GalleryTests(unittest.TestCase):
    def test_gallery_is_encrypted_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "templates.enc"
            key = Fernet.generate_key().decode("ascii")
            gallery = {"student-1": [np.array([1.0, 0.0], dtype=np.float32)]}
            with patch.dict(os.environ, {"ATTENDANCE_TEMPLATE_KEY": key}):
                save_gallery(path, gallery)
                self.assertNotIn(b"student-1", path.read_bytes())
                loaded = load_gallery(path)
            np.testing.assert_allclose(loaded["student-1"][0], [1.0, 0.0])

    def test_ranked_match_uses_cosine_prototypes(self):
        prototypes = normalized_prototypes(
            {
                "a": [np.array([1.0, 0.0]), np.array([0.9, 0.1])],
                "b": [np.array([0.0, 1.0])],
            }
        )
        self.assertEqual("a", rank_matches(np.array([1.0, 0.0]), prototypes)[0][0])

    def test_ranked_match_uses_best_template_not_means(self):
        prototypes = normalized_prototypes(
            {
                "a": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
                "b": [np.array([0.8, 0.6])],
            }
        )
        ranked = rank_matches(np.array([0.0, 1.0]), prototypes)
        self.assertEqual("a", ranked[0][0])


class SplitTests(unittest.TestCase):
    def test_split_keeps_capture_sessions_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            students = root / "students"
            for student_id in ("s1", "s2"):
                for session_index in range(5):
                    session = students / student_id / f"session-{session_index}"
                    session.mkdir(parents=True)
                    (session / "face.jpg").write_bytes(b"fixture")
            manifest_path = root / "splits.json"
            prepare_splits(students, root / "unknown", manifest_path, seed=42)
            splits = load_splits(manifest_path)
            session_sets = {
                split: {
                    Path(record["path"]).parent.name
                    + "/"
                    + Path(record["path"]).parent.parent.name
                    for record in records
                }
                for split, records in splits.items()
            }
            self.assertFalse(session_sets["train"] & session_sets["validation"])
            self.assertFalse(session_sets["train"] & session_sets["test"])
            self.assertFalse(session_sets["validation"] & session_sets["test"])
            self.assertTrue(all(splits[split] for split in ("train", "validation", "test")))


class DeletionTests(unittest.TestCase):
    def test_delete_removes_templates_images_and_stale_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data = root / "data"
            reports = root / "reports"
            (data / "students" / "s1" / "session").mkdir(parents=True)
            (data / "students" / "s2").mkdir(parents=True)
            (data / "students" / "s1" / "session" / "face.jpg").write_bytes(b"fixture")
            gallery_path = data / "templates.enc"
            key = Fernet.generate_key().decode("ascii")
            with patch.dict(os.environ, {"ATTENDANCE_TEMPLATE_KEY": key}):
                save_gallery(
                    gallery_path,
                    {
                        "s1": [np.array([1.0, 0.0])],
                        "s2": [np.array([0.0, 1.0])],
                    },
                )
                reports.mkdir()
                (reports / "threshold.json").write_text("{}", encoding="utf-8")
                (reports / "evaluation.json").write_text("{}", encoding="utf-8")
                (data / "splits.json").write_text(
                    json.dumps(
                        {
                            split: [
                                {"student_id": "s1", "path": "s1-face"},
                                {"student_id": "s2", "path": "s2-face"},
                            ]
                            for split in ("train", "validation", "test")
                        }
                    ),
                    encoding="utf-8",
                )
                (data / "classroom_scenes.json").write_text(
                    json.dumps(
                        {
                            split: [
                                {
                                    "path": "scene.jpg",
                                    "faces": [
                                        {"student_id": "s1", "bbox": [1, 1, 10, 10]},
                                        {"student_id": "s2", "bbox": [20, 1, 10, 10]},
                                    ],
                                }
                            ]
                            for split in ("validation", "test")
                        }
                    ),
                    encoding="utf-8",
                )
                config_path = root / "config.yaml"
                config_path.write_text(
                    "paths:\n"
                    "  templates: data/templates.enc\n"
                    "  students: data/students\n"
                    "  split_manifest: data/splits.json\n"
                    "  classroom_scenes: data/classroom_scenes.json\n"
                    "  reports: reports\n",
                    encoding="utf-8",
                )
                delete_student(str(config_path), "s1", delete_images=True)
                remaining = load_gallery(gallery_path)

            self.assertEqual(set(remaining), {"s2"})
            self.assertFalse((data / "students" / "s1").exists())
            self.assertFalse((reports / "threshold.json").exists())
            split_data = json.loads((data / "splits.json").read_text(encoding="utf-8"))
            self.assertTrue(all(record["student_id"] == "s2"
                                for split in split_data.values() for record in split))
            scene_data = json.loads(
                (data / "classroom_scenes.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(face["student_id"] == "s2"
                    for split in scene_data.values()
                    for scene in split
                    for face in scene["faces"])
            )

    def test_student_id_rejects_path_traversal_and_reserved_unknown(self):
        for student_id in ("../outside", "Unknown"):
            with self.subTest(student_id=student_id):
                with self.assertRaises(ValueError):
                    validate_student_id(student_id)


class RecognitionTests(unittest.TestCase):
    def test_uncertain_match_is_held_for_review(self):
        config = {
            "recognition": {
                "review_margin": 0.04,
                "minimum_identity_margin": 0.03,
            }
        }
        observation = SimpleNamespace(review_reason=None, embedding=np.array([1.0, 0.0]))
        label, _ = classify(
            observation,
            {"student-1": np.array([1.0, 0.0])},
            threshold=0.98,
            config=config,
        )
        self.assertEqual(label, "Review: possible student-1")

    def test_low_quality_match_is_not_forced(self):
        config = {
            "recognition": {"review_margin": 0.04, "minimum_identity_margin": 0.03}
        }
        observation = SimpleNamespace(review_reason="face too small", embedding=np.array([1.0, 0.0]))
        label, _ = classify(
            observation,
            {"student-1": np.array([1.0, 0.0])},
            threshold=0.3,
            config=config,
        )
        self.assertTrue(label.startswith("Review:"))

    def test_threshold_selection_uses_both_known_and_impostor_scores(self):
        records = [
            {"true_id": "a", "embedding": np.array([1.0, 0.0])},
            {"true_id": "Unknown", "embedding": np.array([0.0, 1.0])},
        ]
        threshold, false_positive_rate, true_positive_rate, _ = _scores_for_threshold(
            records,
            {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])},
        )
        self.assertGreaterEqual(threshold, -1.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertEqual(len(false_positive_rate), len(true_positive_rate))

    def test_scene_evaluation_matches_faces_and_reports_misses(self):
        detected = [
            SimpleNamespace(box=(10, 10, 20, 20)),
            SimpleNamespace(box=(70, 70, 20, 20)),
        ]
        annotated = [
            {"bbox": [9, 9, 21, 21]},
            {"bbox": [30, 30, 10, 10]},
        ]
        matches, missed, unmatched = _match_scene_faces(
            detected, annotated, minimum_iou=0.3
        )
        self.assertEqual(matches, {0: 0})
        self.assertEqual(missed, {1})
        self.assertEqual(unmatched, {1})

    def test_scene_records_count_missed_faces_and_unmatched_detections(self):
        detected = [
            FaceObservation((10, 10, 20, 20), np.array([1.0, 0.0]), 0.99, 100, None),
            FaceObservation((80, 80, 15, 15), np.array([0.0, 1.0]), 0.99, 100, None),
        ]
        engine = SimpleNamespace(extract=lambda _image: detected)
        scene = {
            "path": "classroom.jpg",
            "faces": [
                {"student_id": "s1", "bbox": [9, 9, 21, 21]},
                {"student_id": "Unknown", "bbox": [40, 40, 10, 10]},
            ],
        }
        with patch("attendance.evaluate.cv2.imread", return_value=np.zeros((100, 100, 3))):
            records, rejected, ground_truth, matched, unmatched = _extract_scene_records(
                [scene], engine, minimum_iou=0.3
            )
        self.assertEqual(rejected, 0)
        self.assertEqual((ground_truth, matched, unmatched), (2, 1, 1))
        self.assertEqual([record["true_id"] for record in records], ["s1", "Unknown"])
        self.assertTrue(records[1]["undetected"])

    def test_identity_tracker_requires_consecutive_confirmations(self):
        tracker = IdentityTracker(confirmation_frames=2)
        first = tracker.update([
            {"label": "student-1", "score": 0.9, "box": (10, 10, 20, 20)}
        ])
        second = tracker.update([
            {"label": "student-1", "score": 0.91, "box": (11, 10, 20, 20)}
        ])
        self.assertTrue(first[0]["label"].startswith("Review: confirming"))
        self.assertEqual(second[0]["label"], "student-1")

    def test_match_just_below_cutoff_is_held_for_review(self):
        config = {
            "recognition": {
                "review_margin": 0.04,
                "minimum_identity_margin": 0.03,
            }
        }
        embedding = np.array([1.0, 0.0])
        candidate = np.array([0.99, np.sqrt(1 - 0.99**2)])
        label, _ = classify(
            SimpleNamespace(review_reason=None, embedding=embedding),
            {"student-1": candidate},
            threshold=1.0,
            config=config,
        )
        self.assertTrue(label.startswith("Review:"))


if __name__ == "__main__":
    unittest.main()
