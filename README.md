# Classroom face attendance (OpenCV SFace + YuNet)

This is a local, consent-based face recognition starter project. It detects multiple faces in images and video sources, generates aligned SFace embeddings, compares them against enrolled students, and routes uncertain or low-quality matches to manual review. It is not an attendance-recording backend and does not promise any accuracy level.

## What is included

- OpenCV Zoo YuNet face detection and SFace embedding generation; no fine-tuning or custom face model training.
- Enrollment accepts multiple images per student, using **training sessions only**.
- AES-authenticated Fernet encryption for the local embedding gallery; the key is supplied outside the project via an environment variable.
- Session-disjoint train/validation/test splits, validation-only cosine threshold selection, and held-out metrics, ROC/confusion plots, and a misclassification CSV.
- Multi-face image, video, webcam, and RTSP input. Low-quality, borderline, and ambiguous matches are labeled for review rather than silently treated as certain identities.
- A student-template deletion command.

Pretrained recognition models are appropriate for small registration datasets because they already encode general face features learned from large training data. Training a deep face model from a few student images would overfit and would not provide reliable generalization to changes in lighting, expression, pose, camera, or distance. Enrollment here builds identity templates; it does not fine-tune SFace.

## Model and dependency licensing

- [SFace model card and Apache-2.0 model license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
- [YuNet model card and MIT model license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- The Python OpenCV dependency is `opencv-contrib-python`, which provides `FaceDetectorYN` and `FaceRecognizerSF` on Windows. Check upstream licenses and all third-party/model terms for your deployment context; licenses do not by themselves establish that biometric use is lawful.
- Model files are downloaded from the OpenCV Zoo repository at runtime and are not committed here.

## Windows installation

Run these commands in PowerShell from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m attendance.download_models
```

The code requires Python 3.10 or newer. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of activating the environment. Do not install both `opencv-python` and `opencv-contrib-python` in the same environment; this project requires the latter.

Generate a unique key and keep it in a protected secret manager or user-scoped environment variable. **Do not commit or share it.** Losing it makes the encrypted templates unrecoverable.

```powershell
$env:ATTENDANCE_TEMPLATE_KEY = (python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

That command sets the key only for the current PowerShell session. For repeated use, provision the key using your organization's approved secret-management mechanism. Never save the key beside the encrypted template file.

## Dataset layout and collection

Create one folder per student ID and one subfolder per independent capture session. Use varied, consented registration captures (not bursts or adjacent video frames) with different lighting, expressions, and modest pose variation. Aim for several clear, frontal, well-lit images in each session. Images should show one student's face only.

```text
data\
  students\
    student-001\
      registration-day-1\
        front-01.jpg
        front-02.jpg
      registration-day-2\
        side-01.jpg
      registration-day-3\
        glasses.jpg
    student-002\
      session-1\
      session-2\
      session-3\
  unknown\                 # optional; consented non-enrolled people for open-set testing
    guest-01\
      validation-session\
      test-session\
```

For each enrolled student, the split generator requires at least three independent session folders and assigns whole sessions to train, validation, and test (roughly 60/20/20). It never enrolls validation or test images. Flat, unsessioned collections are rejected because random per-image splits can leak near-duplicate frames. For meaningful unknown-person FAR, provide separate, consented non-enrolled identities under `data\unknown`, each with two independent sessions. One unknown session goes to validation and the other to final test. Unknown images never enter the gallery.

Capture genuinely realistic held-out images or video frames for the test sessions. Near-identical frames, repeated crops, or a shared recording split across sets invalidate the result. For small cohorts, the reported metrics are noisy; add independent sessions and identities before relying on them operationally.

### Optional full-classroom scene evaluation

The standard split evaluates a single labeled face per image. For end-to-end crowded-scene validation, create `data\classroom_scenes.json` (the path is configurable) with separate validation and test scenes, manually labeling every visible face. Scene files must be from independent, held-out capture sessions and disjoint from registration images. Example:

```json
{
  "validation": [
    {
      "path": "scenes/validation/classroom-01.jpg",
      "faces": [
        {"student_id": "student-001", "bbox": [120, 80, 65, 72]},
        {"student_id": "Unknown", "bbox": [360, 95, 58, 64]}
      ]
    }
  ],
  "test": [
    {
      "path": "scenes/test/classroom-02.jpg",
      "faces": [
        {"student_id": "student-001", "bbox": [110, 82, 62, 70]},
        {"student_id": "student-002", "bbox": [410, 100, 60, 68]}
      ]
    }
  ]
}
```

Scene paths are resolved relative to the JSON file, and boxes use integer pixel coordinates `[x, y, width, height]`. Labels must match enrolled IDs or `Unknown`. Once this file exists, evaluation uses it for both validation and test instead of the single-face split images, matches detections to annotations by the configured IoU threshold, reports missed-face detection recall and unmatched detections, and sends low-quality detected faces to review. Unannotated detections are excluded from identity metrics, so label every real face in each scene.

## Run the pipeline

1. Prepare the session-level split once. Re-running with a different seed creates a new experiment:

   ```powershell
   python -m attendance.evaluate --prepare-splits --config config.yaml --seed 42
   ```

2. Enroll from the training split only:

   ```powershell
   python -m attendance.enroll --config config.yaml
   ```

   Enroll or refresh a single ID with `--student-id student-001`. Every enrolled student must have at least one usable training image. Images with zero/multiple faces, inadequate size, blur, or low detector confidence are rejected and logged. Improve captures rather than lowering quality filters reflexively.

3. Select the similarity threshold on validation data, then report only the held-out test result:

   ```powershell
   python -m attendance.evaluate --config config.yaml
   ```

   Outputs are in `reports\`: `evaluation.json`, `threshold.json`, `validation_roc.png`, `test_confusion_matrix.png`, and `test_misclassified.csv`. Recognition uses the saved validation threshold. It is not reselected on the final test set. If no non-enrolled faces are provided in the validation/test images or scene manifest, real unknown-person FAR is explicitly `null`; a separate impostor-pair proxy is reported instead.

4. Recognize an image, video, webcam, or configured RTSP stream:

   ```powershell
   python -m attendance.recognize --source "classroom.jpg" --config config.yaml --output "annotated.jpg"
   python -m attendance.recognize --source "lesson.mp4" --config config.yaml --output "annotated.mp4" --no-display
   python -m attendance.recognize --source 0 --config config.yaml
   $env:ATTENDANCE_RTSP_URL = "<retrieve-from-your-approved-secret-manager>"
   python -m attendance.recognize --source $env:ATTENDANCE_RTSP_URL --config config.yaml --no-display
   ```

   A numeric source selects a local camera index. RTSP support depends on the OpenCV video I/O/FFmpeg build and camera authentication/network policy; verify access in your environment. Retrieve RTSP credentials through an approved secret manager and avoid embedding them in shared command history or logs.    Processing uses configurable frame skipping and temporal identity confirmation for video streams. Tracking does not make a weak match safe and does not replace independent validation; crowded or distant-room imagery can still exceed the detector's useful resolution. The included YuNet model card describes face sizes around 10–300 pixels, but this is not an identification guarantee. The project further holds faces below the configured 60-pixel minimum for review.
5. Delete an enrolled student's templates; optionally also permanently delete their source images:

   ```powershell
   python -m attendance.delete_student --student-id student-001 --config config.yaml
   python -m attendance.delete_student --student-id student-001 --delete-images --config config.yaml
   ```

   The command asks for confirmation unless `--yes` is provided. It also removes the student's entries from split and classroom-scene manifests and invalidates prior evaluation reports/thresholds because they no longer describe the current gallery. Backups, copies, exports, and logs must be handled under your retention policy separately.

## Configuration and interpretation

Edit `config.yaml` to tune detector confidence, non-maximum suppression, minimum face size, blur cutoff, initial fallback threshold, manual-review margin, video frame processing interval, temporal confirmation frames, and model/data paths. The initial threshold is only a fallback; evaluate first. Identity ranking uses the maximum cosine similarity across all enrolled templates for a student rather than treating their mean as the only template. The runtime score is cosine similarity, not a calibrated probability.
Threshold selection maximizes Youden's J (balanced true-positive and false-positive rates) on validation genuine and impostor/unknown scores. Test reports include precision/recall/F1 by identity, confusion matrix, known false-rejection rate (FRR), unknown false-acceptance rate (FAR) when unknown test faces exist, a cross-identity impostor FAR proxy, top-1 identification accuracy, review count, and excluded-image counts. Inspect the misclassification CSV and held-out captures; a good closed-set top-1 score alone does not establish safe open-set recognition. Uncertain matches are intentionally not auto-assigned; they require human review and an independent attendance check.

Quality checks cover face size, detector confidence, and a simple blur measure. They do not reliably identify occlusion, demographic bias, spoofing, or all pose and lighting problems. A low-quality face is sent for review, not force-matched. Improve acquisition and run new, session-disjoint validation/test experiments before changing thresholds or quality settings.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover encrypted template storage, match ranking, session leakage prevention, uncertain-match handling, and threshold-selection inputs. They do not establish recognition accuracy; that requires a consented, representative, held-out classroom dataset.

## Responsible use and limitations

Obtain informed consent and any required guardian/organizational authorization before collecting face images, especially for minors. Offer an accessible non-biometric attendance alternative. Restrict image/template access, secure the encryption key, define retention, prevent reuse for unrelated purposes, and have a human verify attendance and all uncertain decisions. Do not use outputs for disciplinary, grading, access-denial, or other consequential decisions without appropriate legal review, independent validation, and safeguards. This example is not a liveness detector, anti-spoofing system, production security control, or legal-compliance assessment.
