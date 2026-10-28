from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)
KEY_ENVIRONMENT_VARIABLE = "ATTENDANCE_TEMPLATE_KEY"


def _cipher() -> Fernet:
    key = os.environ.get(KEY_ENVIRONMENT_VARIABLE)
    if not key:
        raise RuntimeError(
            f"Set {KEY_ENVIRONMENT_VARIABLE} to a securely stored Fernet key. "
            "Generate one with `python -c \"from cryptography.fernet import "
            "Fernet; print(Fernet.generate_key().decode())\"`."
        )
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{KEY_ENVIRONMENT_VARIABLE} is not a valid Fernet key.") from error


def load_gallery(path: str | Path) -> dict[str, list[np.ndarray]]:
    gallery_path = Path(path)
    if not gallery_path.is_file():
        return {}
    try:
        payload = _cipher().decrypt(gallery_path.read_bytes())
        decoded = json.loads(payload)
    except InvalidToken as error:
        raise ValueError(
            f"Could not decrypt {gallery_path}; check the template key."
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(f"Invalid encrypted gallery structure: {gallery_path}")
    gallery = {}
    for student_id, vectors in decoded.items():
        if not isinstance(student_id, str) or not isinstance(vectors, list) or not vectors:
            raise ValueError(f"Invalid templates for student {student_id!r}.")
        embeddings = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        if any(vector.ndim != 1 or not np.isfinite(vector).all() for vector in embeddings):
            raise ValueError(f"Invalid embedding vector for student {student_id!r}.")
        gallery[student_id] = embeddings
    return gallery


def save_gallery(path: str | Path, gallery: dict[str, list[np.ndarray]]) -> None:
    gallery_path = Path(path)
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            student_id: [vector.astype(float).tolist() for vector in vectors]
            for student_id, vectors in sorted(gallery.items())
        },
        separators=(",", ":"),
    ).encode("utf-8")
    encrypted = _cipher().encrypt(payload)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=gallery_path.parent, prefix=".templates-", delete=False
        ) as temporary:
            temporary.write(encrypted)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, gallery_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    logger.info("Encrypted gallery saved to %s", gallery_path)


def normalized_prototypes(gallery: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    prototypes = {}
    for student_id, vectors in gallery.items():
        matrix = np.stack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError(f"Cannot build valid templates for {student_id!r}.")
        prototypes[student_id] = matrix / norms
    return prototypes


def rank_matches(
    embedding: np.ndarray, prototypes: dict[str, np.ndarray]
) -> list[tuple[str, float]]:
    if not prototypes:
        raise ValueError("No enrolled student templates are available.")
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 1 or not np.isfinite(embedding).all():
        raise ValueError("Query embedding must be a finite one-dimensional vector.")
    embedding_norm = float(np.linalg.norm(embedding))
    if embedding_norm == 0:
        raise ValueError("Query embedding must have a non-zero norm.")
    embedding = embedding / embedding_norm
    scores = []
    for student_id, prototype in prototypes.items():
        candidate_templates = np.asarray(prototype, dtype=np.float32)
        if candidate_templates.ndim == 1:
            candidate_templates = candidate_templates.reshape(1, -1)
        elif candidate_templates.ndim != 2:
            raise ValueError(f"Invalid prototype shape for {student_id!r}.")
        template_norms = np.linalg.norm(candidate_templates, axis=1, keepdims=True)
        if np.any(template_norms == 0) or not np.isfinite(template_norms).all():
            raise ValueError(f"Invalid prototype for {student_id!r}.")
        normalized = candidate_templates / template_norms
        score = float(np.max(normalized @ embedding))
        scores.append((student_id, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def invalidate_evaluation_reports(report_directory: str | Path) -> None:
    for filename in (
        "threshold.json",
        "evaluation.json",
        "validation_roc.png",
        "test_confusion_matrix.png",
        "test_misclassified.csv",
    ):
        (Path(report_directory) / filename).unlink(missing_ok=True)
