from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2

from attendance.config import load_config

logger = logging.getLogger(__name__)
MODELS = {
    "detector_model": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "recognizer_model": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def download_models(config_path: str | Path) -> None:
    config, _ = load_config(config_path)
    for name, url in MODELS.items():
        destination = Path(config["paths"][name])
        if destination.is_file():
            logger.info("Model already exists: %s", destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            logger.info("Downloading %s", url)
            urllib.request.urlretrieve(url, temporary)
            if temporary.stat().st_size < 10_000:
                raise ValueError(
                    f"Downloaded model is unexpectedly small ({temporary.stat().st_size} bytes)."
                )
            network = cv2.dnn.readNetFromONNX(str(temporary))
            if network.empty():
                raise ValueError(f"Downloaded ONNX model could not be loaded: {url}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    logger.info("Required face models are ready.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download the OpenCV Zoo ONNX models.")
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    download_models(arguments.config)
