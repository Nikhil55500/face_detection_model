from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

from attendance.config import load_config
from attendance.face_engine import FaceEngine
from attendance.storage import load_gallery, normalized_prototypes, rank_matches

logger = logging.getLogger(__name__)


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    left, top = max(x1, x2), max(y1, y2)
    right, bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


class IdentityTracker:
    def __init__(self, confirmation_frames: int = 3):
        self.confirmation_frames = max(1, confirmation_frames)
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, results: list[dict]) -> list[dict]:
        matched_tracks = set()
        updated = []
        for result in results:
            label = result["label"]
            candidates = []
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                overlap = _box_iou(result["box"], track["box"])
                if overlap >= 0.3:
                    candidates.append((overlap, track_id))
            track_id = max(candidates)[1] if candidates else None
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {"box": result["box"], "label": label, "count": 0}
            track = self.tracks[track_id]
            matched_tracks.add(track_id)
            if track["label"] == label and label != "Unknown" and not label.startswith("Review:"):
                track["count"] += 1
            else:
                track["label"] = label
                track["count"] = 1 if label == "Unknown" or label.startswith("Review:") else 0
            track["box"] = result["box"]
            if (
                label != "Unknown"
                and not label.startswith("Review:")
                and track["count"] < self.confirmation_frames
            ):
                result = dict(result, label=f"Review: confirming {label}")
            updated.append(result)
        active = set(self.tracks)
        stale = active - matched_tracks
        for track_id in stale:
            del self.tracks[track_id]
        return updated


def load_threshold(config: dict) -> float:
    threshold_path = Path(config["paths"]["threshold_report"])
    if threshold_path.is_file():
        threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    else:
        threshold = float(config["recognition"]["initial_threshold"])
        logger.warning("No validation threshold found; using configured initial threshold %.3f.",
                       threshold)
    if not -1 <= threshold <= 1:
        raise ValueError("Cosine similarity threshold must be between -1 and 1.")
    return threshold


def classify(observation, prototypes, threshold: float, config: dict) -> tuple[str, float]:
    if observation.review_reason:
        return f"Review: {observation.review_reason}", 0.0
    ranked = rank_matches(observation.embedding, prototypes)
    best_id, best_score = ranked[0]
    review_margin = float(config["recognition"]["review_margin"])
    if best_score < threshold - review_margin:
        return "Unknown", best_score
    if best_score < threshold + review_margin:
        return f"Review: possible {best_id}", best_score
    recognition = config["recognition"]
    if len(ranked) > 1 and best_score - ranked[1][1] < float(
        recognition["minimum_identity_margin"]
    ):
        return f"Review: ambiguous ({best_id}, {ranked[1][0]})", best_score
    return best_id, best_score


def annotate_frame(frame, engine, prototypes, threshold, config):
    results = []
    for observation in engine.extract(frame):
        label, score = classify(observation, prototypes, threshold, config)
        x, y, width, height = observation.box
        color = (0, 165, 255) if label.startswith("Review:") else (
            (0, 180, 0) if label != "Unknown" else (0, 0, 255)
        )
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            f"{label} [{score:.2f}]",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        results.append({"label": label, "score": score, "box": observation.box})
    return frame, results


def _source_value(source: str):
    return int(source) if source.isdecimal() else source


def recognize(
    source: str,
    config_path: str,
    output_path: str | None = None,
    display: bool = True,
) -> None:
    config, _ = load_config(config_path)
    gallery = load_gallery(config["paths"]["templates"])
    if not gallery:
        raise ValueError("The encrypted gallery is empty. Enroll students first.")
    prototypes = normalized_prototypes(gallery)
    engine = FaceEngine(config)
    threshold = load_threshold(config)
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise ValueError(f"Could not read image: {source_path}")
        annotated, results = annotate_frame(frame, engine, prototypes, threshold, config)
        if output_path:
            if not cv2.imwrite(output_path, annotated):
                raise OSError(f"Could not write output image: {output_path}")
        if display:
            cv2.imshow("Classroom attendance - review uncertain matches", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        logger.info("Image results: %s", json.dumps(results))
        return

    capture = cv2.VideoCapture(_source_value(source), cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        raise OSError("Could not open the requested video source.")
    writer = None
    recognition_config = config.get("recognition", {})
    process_every_n_frames = max(1, int(recognition_config.get("process_every_n_frames", 1)))
    tracker = IdentityTracker(int(recognition_config.get("temporal_confirm_frames", 3)))
    frame_number = 0
    last_results = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            if frame_number % process_every_n_frames == 1:
                annotated, results = annotate_frame(frame, engine, prototypes, threshold, config)
                results = tracker.update(results)
                last_results = results
            else:
                annotated = frame
                for result in last_results:
                    x, y, width, height = result["box"]
                    label, score = result["label"], result["score"]
                    cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 165, 255), 2)
                    cv2.putText(annotated, f"{label} [{score:.2f}]", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
                results = last_results
            if output_path and writer is None:
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"Could not create output video: {output_path}")
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("Classroom attendance - review uncertain matches", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            logger.debug("Frame results: %s", results)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize faces in an image or video stream.")
    parser.add_argument("--source", required=True, help="Image/video path, camera index, or RTSP URL.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", help="Optional annotated image or MP4 output.")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    recognize(arguments.source, arguments.config, arguments.output, not arguments.no_display)


if __name__ == "__main__":
    main()
