from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)

from attendance.config import load_config
from attendance.dataset import load_splits, prepare_splits
from attendance.face_engine import FaceEngine
from attendance.storage import load_gallery, normalized_prototypes, rank_matches

logger = logging.getLogger(__name__)


def _extract_records(records, engine):
    results = []
    rejected = 0
    for record in records:
        image_path = Path(record["path"])
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Unreadable evaluation image: %s", image_path)
            rejected += 1
            continue
        faces = engine.extract(image)
        if len(faces) != 1 or faces[0].review_reason:
            reason = "no unique face" if len(faces) != 1 else faces[0].review_reason
            logger.warning("Excluded evaluation image (%s): %s", reason, image_path)
            rejected += 1
            continue
        results.append(
            {
                "path": str(image_path),
                "true_id": record["student_id"],
                "embedding": faces[0].embedding,
            }
        )
    return results, rejected


def _box_iou(first: tuple[int, int, int, int], second: list[int]) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def _match_scene_faces(detected, annotated_faces, minimum_iou: float):
    candidate_pairs = sorted(
        (
            (_box_iou(observation.box, annotation["bbox"]), detection_index, face_index)
            for detection_index, observation in enumerate(detected)
            for face_index, annotation in enumerate(annotated_faces)
        ),
        reverse=True,
    )
    detection_to_face = {}
    used_faces = set()
    used_detections = set()
    for overlap, detection_index, face_index in candidate_pairs:
        if overlap < minimum_iou:
            break
        if detection_index in used_detections or face_index in used_faces:
            continue
        detection_to_face[detection_index] = face_index
        used_detections.add(detection_index)
        used_faces.add(face_index)
    missed_faces = set(range(len(annotated_faces))) - used_faces
    unmatched_detections = set(range(len(detected))) - used_detections
    return detection_to_face, missed_faces, unmatched_detections


def _load_scene_manifest(
    path: str | Path, gallery_ids: set[str], excluded_image_paths: set[str]
):
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        split not in manifest for split in ("validation", "test")
    ):
        raise ValueError("Scene manifest must contain validation and test arrays.")
    scenes = {}
    for split in ("validation", "test"):
        scenes[split] = []
        for scene in manifest[split]:
            scene_path = Path(scene["path"])
            if not scene_path.is_absolute():
                scene_path = (manifest_path.parent / scene_path).resolve()
            faces = scene.get("faces")
            if not isinstance(faces, list):
                raise ValueError(f"Scene {scene_path} must have a faces array.")
            for face in faces:
                student_id = face.get("student_id")
                bbox = face.get("bbox")
                if student_id != "Unknown" and student_id not in gallery_ids:
                    raise ValueError(
                        f"Scene {scene_path} references unenrolled identity {student_id!r}."
                    )
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(not isinstance(value, int) for value in bbox)
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    raise ValueError(
                        f"Scene {scene_path} has an invalid [x, y, width, height] box."
                    )
            scenes[split].append({"path": str(scene_path), "faces": faces})
    validation_paths = {scene["path"] for scene in scenes["validation"]}
    test_paths = {scene["path"] for scene in scenes["test"]}
    if validation_paths & test_paths:
        raise ValueError("A classroom scene cannot appear in both validation and test.")
    if (validation_paths | test_paths) & excluded_image_paths:
        raise ValueError("Classroom scene images cannot reuse any dataset-split image.")
    return scenes


def _extract_scene_records(scenes, engine, minimum_iou: float):
    records = []
    rejected_scenes = 0
    ground_truth_count = 0
    matched_ground_truth_count = 0
    unmatched_detection_count = 0
    for scene in scenes:
        image_path = Path(scene["path"])
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Unreadable classroom scene: %s", image_path)
            rejected_scenes += 1
            continue
        annotated_faces = scene["faces"]
        ground_truth_count += len(annotated_faces)
        observations = engine.extract(image)
        detection_to_face, missed_faces, unmatched_detections = _match_scene_faces(
            observations, annotated_faces, minimum_iou
        )
        matched_ground_truth_count += len(detection_to_face)
        unmatched_detection_count += len(unmatched_detections)
        for detection_index, observation in enumerate(observations):
            face_index = detection_to_face.get(detection_index)
            if face_index is None:
                continue
            student_id = annotated_faces[face_index]["student_id"]
            if observation.review_reason:
                logger.warning("Low-quality face in classroom scene (%s).",
                               observation.review_reason)
            records.append(
                {
                    "path": f"{image_path}#face-{detection_index}",
                    "true_id": student_id,
                    "embedding": observation.embedding,
                    "review_reason": observation.review_reason,
                }
            )
        for face_index in sorted(missed_faces):
            face = annotated_faces[face_index]
            records.append(
                {
                    "path": f"{image_path}#missed-face-{face_index}",
                    "true_id": face["student_id"],
                    "embedding": None,
                    "undetected": True,
                }
            )
    return (
        records,
        rejected_scenes,
        ground_truth_count,
        matched_ground_truth_count,
        unmatched_detection_count,
    )


def _scores_for_threshold(records, prototypes):
    scores, labels = [], []
    for record in records:
        if record.get("undetected") or record.get("review_reason"):
            continue
        ranked = rank_matches(record["embedding"], prototypes)
        if record["true_id"] == "Unknown":
            scores.append(ranked[0][1])
            labels.append(0)
            continue
        matching = next(
            (score for student_id, score in ranked if student_id == record["true_id"]),
            None,
        )
        if matching is None:
            raise ValueError(f"No enrolled template for {record['true_id']!r}.")
        scores.append(matching)
        labels.append(1)
        impostor_scores = [
            score for student_id, score in ranked if student_id != record["true_id"]
        ]
        if impostor_scores:
            scores.append(max(impostor_scores))
            labels.append(0)
    if not any(labels) or all(labels):
        raise ValueError(
            "Validation must contain known faces and at least one unknown/impostor score."
        )
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    valid_indices = [
        index for index, threshold in enumerate(thresholds)
        if np.isfinite(threshold) and -1.0 <= threshold <= 1.0
    ]
    if not valid_indices:
        raise ValueError("Validation scores did not yield a finite cosine threshold.")
    best_index = max(
        valid_indices,
        key=lambda index: (
            true_positive_rate[index] - false_positive_rate[index],
            true_positive_rate[index],
            -false_positive_rate[index],
        ),
    )
    threshold = float(thresholds[best_index])
    return threshold, false_positive_rate, true_positive_rate, thresholds


def _predict(records, prototypes, threshold, config):
    predicted, details = [], []
    review_margin = float(config["recognition"]["review_margin"])
    identity_margin = float(config["recognition"]["minimum_identity_margin"])
    for record in records:
        if record.get("undetected"):
            predicted.append("Undetected")
            details.append(
                {
                    "path": record["path"],
                    "true_id": record["true_id"],
                    "predicted": "Undetected",
                    "similarity": None,
                    "best_candidate": None,
                }
            )
            continue
        if record.get("review_reason"):
            predicted.append("Review")
            details.append(
                {
                    "path": record["path"],
                    "true_id": record["true_id"],
                    "predicted": "Review",
                    "similarity": None,
                    "best_candidate": None,
                }
            )
            continue
        ranked = rank_matches(record["embedding"], prototypes)
        best_id, best_score = ranked[0]
        if best_score < threshold - review_margin:
            label = "Unknown"
        elif best_score < threshold + review_margin:
            label = "Review"
        elif len(ranked) > 1 and best_score - ranked[1][1] < identity_margin:
            label = "Review"
        else:
            label = best_id
        predicted.append(label)
        details.append(
            {
                "path": record["path"],
                "true_id": record["true_id"],
                "predicted": label,
                "similarity": best_score,
                "best_candidate": best_id,
            }
        )
    return predicted, details


def _plot_roc(path, false_positive_rate, true_positive_rate):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label="Validation")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False acceptance rate", ylabel="True acceptance rate",
             title="Validation ROC (known vs impostor/unknown)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_confusion(path, matrix, labels):
    size = max(6, min(16, len(labels) * 0.6))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True identity",
        xlabel="Predicted identity",
        title="Test-set confusion matrix (Review counted as Unknown)",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate(config_path: str, prepare: bool, seed: int) -> None:
    config, _ = load_config(config_path)
    if prepare:
        prepare_splits(
            config["paths"]["students"],
            config["paths"]["unknown"],
            config["paths"]["split_manifest"],
            seed,
        )
        return
    splits = load_splits(config["paths"]["split_manifest"])
    gallery = load_gallery(config["paths"]["templates"])
    if not gallery:
        raise ValueError("No templates available; run enrollment on the training split first.")
    prototypes = normalized_prototypes(gallery)
    engine = FaceEngine(config)
    scene_manifest_path = Path(config["paths"]["classroom_scenes"])
    validation_ground_truth = None
    test_ground_truth = None
    validation_unmatched_detections = None
    test_unmatched_detections = None
    if scene_manifest_path.is_file():
        excluded_image_paths = {
            str(Path(record["path"]).resolve())
            for split_records in splits.values()
            for record in split_records
        }
        scenes = _load_scene_manifest(
            scene_manifest_path, set(gallery), excluded_image_paths
        )
        scene_iou = float(config["detection"]["scene_match_iou"])
        (
            validation,
            rejected_validation,
            validation_ground_truth,
            validation_matched,
            validation_unmatched_detections,
        ) = _extract_scene_records(scenes["validation"], engine, scene_iou)
        (
            test,
            rejected_test,
            test_ground_truth,
            test_matched,
            test_unmatched_detections,
        ) = _extract_scene_records(scenes["test"], engine, scene_iou)
    else:
        validation, rejected_validation = _extract_records(splits["validation"], engine)
        test, rejected_test = _extract_records(splits["test"], engine)
        validation_matched = test_matched = None
    if not validation or not test:
        raise ValueError("Validation and test splits must each contain usable faces.")

    threshold, fpr, tpr, roc_thresholds = _scores_for_threshold(validation, prototypes)
    _, validation_details = _predict(validation, prototypes, threshold, config)
    test_predictions, test_details = _predict(test, prototypes, threshold, config)
    report_dir = Path(config["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(gallery) | {"Unknown"})
    true_labels = [record["true_id"] for record in test]
    scored_predictions = ["Unknown" if prediction == "Review" else prediction
                          for prediction in test_predictions]
    if any(prediction == "Undetected" for prediction in test_predictions):
        labels = sorted(set(labels) | {"Undetected"})
    matrix = confusion_matrix(true_labels, scored_predictions, labels=labels)
    metrics = classification_report(
        true_labels, scored_predictions, labels=labels, output_dict=True, zero_division=0
    )
    known_results = [
        (record, prediction)
        for record, prediction in zip(test, test_predictions)
        if record["true_id"] != "Unknown"
    ]
    unknown_results = [
        prediction
        for record, prediction in zip(test, test_predictions)
        if record["true_id"] == "Unknown"
        and not record.get("undetected")
        and not record.get("review_reason")
    ]
    known_false_rejections = sum(
        prediction in {"Unknown", "Review", "Undetected"}
        for _, prediction in known_results
    )
    unknown_false_acceptances = sum(
        prediction not in {"Unknown", "Review"} for prediction in unknown_results
    )
    test_impostor_scores = []
    for record in test:
        if (
            record["true_id"] == "Unknown"
            or record.get("undetected")
            or record.get("review_reason")
        ):
            continue
        test_impostor_scores.extend(
            score
            for student_id, score in rank_matches(record["embedding"], prototypes)
            if student_id != record["true_id"]
        )
    misclassified = [item for item in test_details if item["true_id"] != item["predicted"]]
    report = {
        "threshold": threshold,
        "threshold_selection": "Validation ROC Youden J (balanced known/impostor decision)",
        "test_metrics": {
            "classification_report": metrics,
            "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
            "known_false_rejection_rate": (
                known_false_rejections / len(known_results) if known_results else None
            ),
            "unknown_false_acceptance_rate": (
                unknown_false_acceptances / len(unknown_results) if unknown_results else None
            ),
            "test_unknown_face_count": len(unknown_results),
            "impostor_pairwise_far_proxy": (
                sum(score >= threshold for score in test_impostor_scores)
                / len(test_impostor_scores)
                if test_impostor_scores else None
            ),
            "closed_set_top1_identification_accuracy": (
                sum(
                    rank_matches(record["embedding"], prototypes)[0][0] == record["true_id"]
                    for record, _ in known_results
                    if not record.get("undetected") and not record.get("review_reason")
                )
                / sum(
                    not record.get("undetected") and not record.get("review_reason")
                    for record, _ in known_results
                )
                if any(
                    not record.get("undetected") and not record.get("review_reason")
                    for record, _ in known_results
                )
                else None
            ),
            "review_count": test_predictions.count("Review"),
            "usable_test_faces": sum(not record.get("undetected") for record in test),
            "excluded_test_images": rejected_test,
            "ground_truth_face_count": test_ground_truth,
            "detected_ground_truth_face_count": test_matched,
            "missed_ground_truth_face_count": (
                test_ground_truth - test_matched if test_ground_truth is not None else None
            ),
            "face_detection_recall": (
                test_matched / test_ground_truth
                if test_ground_truth is not None and test_ground_truth
                else None
            ),
            "unmatched_detections": test_unmatched_detections,
            "scene_detection_precision": (
                test_matched / (test_matched + test_unmatched_detections)
                if test_matched is not None
                and test_matched + test_unmatched_detections > 0
                else None
            ),
        },
        "validation": {
            "usable_faces": len(validation),
            "excluded_images": rejected_validation,
            "ground_truth_face_count": validation_ground_truth,
            "detected_ground_truth_face_count": validation_matched,
            "unmatched_detections": validation_unmatched_detections,
            "details": validation_details,
        },
        "limitations": (
            "Unknown FAR is only measured against real, non-enrolled people if unknown "
            "capture sessions were supplied. Otherwise it is null; impostor-pair FAR "
            "is a proxy, not a real-world unknown-person estimate."
        ),
    }
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold}, indent=2), encoding="utf-8"
    )
    _plot_roc(report_dir / "validation_roc.png", fpr, tpr)
    _plot_confusion(report_dir / "test_confusion_matrix.png", matrix, labels)
    with (report_dir / "test_misclassified.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=test_details[0].keys())
        writer.writeheader()
        writer.writerows(misclassified)
    logger.info(
        "Test results saved to %s; threshold %.4f, test faces %d, unknown test faces %d.",
        report_dir, threshold, len(test), len(unknown_results),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare session splits or evaluate recognition.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prepare-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluate(arguments.config, arguments.prepare_splits, arguments.seed)


if __name__ == "__main__":
    main()
