from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st

from attendance.config import load_config
from attendance.download_models import download_models
from attendance.face_engine import FaceEngine
from attendance.identifiers import validate_student_id
from attendance.recognize import annotate_frame, load_threshold
from attendance.storage import load_gallery, normalized_prototypes, save_gallery


CONFIG_PATH = "config.yaml"


@st.cache_resource
def get_engine() -> tuple[dict, FaceEngine]:
    config, _ = load_config(CONFIG_PATH)
    download_models(CONFIG_PATH)
    return config, FaceEngine(config)


def decode_upload(uploaded_file) -> np.ndarray | None:
    bytes_data = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    return cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)


def draw_enrollment_preview(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    preview = image.copy()
    for x, y, width, height in boxes:
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 180, 0), 2)
    return preview


st.set_page_config(page_title="Classroom Face Attendance", layout="wide")
st.title("Classroom Face Attendance")
st.warning(
    "Biometric processing requires informed consent and appropriate legal authorization. "
    "Protect the app with authentication; do not expose this demo publicly."
)

if not os.environ.get("ATTENDANCE_TEMPLATE_KEY"):
    st.error("ATTENDANCE_TEMPLATE_KEY is not configured. Set it before using enrollment or recognition.")
    st.stop()

try:
    config, engine = get_engine()
except Exception as error:
    st.error(f"Application setup failed: {error}")
    st.stop()

gallery_path = config["paths"]["templates"]

enroll_tab, recognize_tab = st.tabs(["Enroll student", "Recognize faces"])

with enroll_tab:
    st.subheader("Enroll a student")
    student_id = st.text_input(
        "Student ID",
        placeholder="student-001",
        help="1-64 safe characters: letters, digits, dot, underscore, or hyphen.",
    )
    uploads = st.file_uploader(
        "Upload several clear, consented registration images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if st.button("Create or update templates", type="primary", disabled=not uploads):
        try:
            validate_student_id(student_id)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        embeddings = []
        previews = []
        rejected = 0
        for uploaded in uploads:
            image = decode_upload(uploaded)
            if image is None:
                rejected += 1
                continue
            observations = engine.extract(image)
            if len(observations) != 1 or observations[0].review_reason:
                rejected += 1
                previews.append(draw_enrollment_preview(
                    image,
                    [observation.box for observation in observations],
                ))
                continue
            observation = observations[0]
            embeddings.append(observation.embedding)
            previews.append(draw_enrollment_preview(image, [observation.box]))

        if not embeddings:
            st.error("No usable enrollment image was found. Use clear, single-face images.")
        else:
            try:
                gallery = load_gallery(gallery_path)
                gallery[student_id] = embeddings
                save_gallery(gallery_path, gallery)
                st.success(
                    f"Saved {len(embeddings)} template(s) for {student_id}; "
                    f"rejected {rejected} file(s)."
                )
            except Exception as error:
                st.error(str(error))

        if previews:
            columns = st.columns(min(3, len(previews)))
            for index, preview in enumerate(previews):
                columns[index % len(columns)].image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

with recognize_tab:
    st.subheader("Recognize faces")
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    uploaded = st.camera_input("Take a photo") if source == "Camera" else st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is not None:
        image = decode_upload(uploaded)
        if image is None:
            st.error("The uploaded image could not be decoded.")
        else:
            try:
                gallery = load_gallery(gallery_path)
                prototypes = normalized_prototypes(gallery)
                threshold = load_threshold(config)
                annotated, results = annotate_frame(
                    image.copy(),
                    engine,
                    prototypes,
                    threshold,
                    config,
                )
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                if results:
                    st.dataframe(results, use_container_width=True, hide_index=True)
                else:
                    st.info("No faces were detected.")
            except Exception as error:
                st.error(str(error))
