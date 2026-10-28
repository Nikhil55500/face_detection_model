from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    box: tuple[int, int, int, int]
    embedding: np.ndarray
    detection_score: float
    blur_score: float
    review_reason: str | None


class FaceEngine:
    def __init__(self, config: dict):
        detection = config["detection"]
        detector_path = Path(config["paths"]["detector_model"])
        recognizer_path = Path(config["paths"]["recognizer_model"])
        for model_path in (detector_path, recognizer_path):
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Required ONNX model not found: {model_path}. "
                    "Run `python -m attendance.download_models`."
                )
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_path),
            "",
            (320, 320),
            float(detection["confidence"]),
            float(detection["nms_threshold"]),
            int(detection["top_k"]),
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")
        self.minimum_face_size = int(detection["minimum_face_size"])
        self.minimum_blur_score = float(detection["minimum_blur_score"])
        self.minimum_detection_score = float(detection["minimum_detection_score"])

    def extract(self, image: np.ndarray) -> list[FaceObservation]:
        if image is None or image.size == 0 or image.ndim != 3:
            raise ValueError("Expected a non-empty BGR image.")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, detected = self.detector.detect(image)
        if detected is None:
            return []

        observations = []
        for face in detected:
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("SFace produced an invalid embedding.")
            embedding = feature / norm
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            detection_score = float(face[-1])
            reasons = []
            if min(box_width, box_height) < self.minimum_face_size:
                reasons.append("face too small")
            if blur_score < self.minimum_blur_score:
                reasons.append("blurred face")
            if detection_score < self.minimum_detection_score:
                reasons.append("low detection confidence")
            observations.append(
                FaceObservation(
                    box=(x, y, box_width, box_height),
                    embedding=embedding,
                    detection_score=detection_score,
                    blur_score=blur_score,
                    review_reason=", ".join(reasons) if reasons else None,
                )
            )
        return observations
