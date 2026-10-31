# Deployment Guide

This project can be deployed as a containerized Streamlit application for enrollment and
image-based face recognition. It does not expose a public webcam device or the local
CLI tools unless they are added separately. Treat all biometric data as sensitive personal
data, require informed consent, restrict access, and use a deployment environment with
TLS and authentication.

## Important privacy and security requirements

- Never commit `ATTENDANCE_TEMPLATE_KEY`, `data/templates.enc`, or student source images.
- Generate a unique key per environment. Losing the key makes encrypted templates unrecoverable.
- Do not expose the app publicly without authentication and HTTPS.
- Store the template key and template volume through approved secret and backup systems.
- The Streamlit app is suitable for a controlled demo or internal deployment, not as evidence
  of perfect identity verification.
- Use a persistent volume for `/app/data`, because enrollment writes to `data/templates.enc`.

## Prerequisites

- Docker Engine with Docker Compose
- A host with at least 2 GB RAM
- An HTTPS/TLS reverse proxy such as Caddy, nginx, or a cloud load balancer
- The two bundled ONNX models are present in `models/` before the Docker build
- An organization-approved authentication layer

Generate a Fernet key locally:

```powers
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

PowerShell command examples below use PowerShell syntax.

## Deploy with Docker

From the project directory:

```powershell
docker build --tag classroom-face-attendance:1.0.0 .
```

Create the persistent data directory and run the app:

```powershell
docker volume create attendance-data
docker run --detach `
  --name classroom-face-attendance `
  --restart unless-stopped `
  --publish 127.0.0.1:8501:8501 `
  --env "ATTENDANCE_TEMPLATE_KEY=<PASTE_YOUR_FERNET_KEY>" `
  --mount type=volume,source=attendance-data,target=/app/data `
  classroom-face-attendance:1.0.0
```

Verify the container:

```powershell
docker logs classroom-face-attendance
docker inspect --format="{{.State.Health.Status}}" classroom-face-attendance
curl.exe http://127.0.0.1:8501/_stcore/health
```

Access the app only through your HTTPS reverse proxy. The direct bind above is for local
host use and is not externally reachable.

## Docker Compose

Create a `.env` file next to the project:

```dotenv
ATTENDANCE_TEMPLATE_KEY=<PASTE_YOUR_FERNET_KEY>
```

Create `docker-compose.yml`:

```yaml
services:
  attendance:
    build: .
    image: classroom-face-attendance:1.0.0
    restart: unless-stopped
    ports:
      - "127.0.0.1:8501:8501"
    environment:
      ATTENDANCE_TEMPLATE_KEY: "${ATTENDANCE_TEMPLATE_KEY}"
    volumes:
      - attendance-data:/app/data

volumes:
  attendance-data:
```

Start and verify:

```powershell
docker compose up --build --detach
docker compose ps
docker compose logs --follow attendance
curl.exe http://127.0.0.1:8501/_stcore/health
```

Do not commit `.env`.

## Cloud deployment

The container can be deployed to Azure Container Apps, AWS App Runner, Render, Railway,
Google Cloud Run, or another platform that supports Docker and persistent storage.
Configure:

- Container port: `8501`
- Health check path: `/_stcore/health`
- Command: `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`
- Secret: `ATTENDANCE_TEMPLATE_KEY`
- Persistent mount or managed backup: `/app/data`
- Minimum resources: 1 vCPU and 2 GB memory
- Timeouts: at least 60 seconds for image upload processing
- Websocket support: enabled for Streamlit
- TLS: enabled by the platform or reverse proxy

If the platform does not support mounted storage, use an external file store and update the
`config.yaml` paths before production deployment. Do not rely on an ephemeral container
filesystem for enrolled templates.

## Reverse proxy example

The Streamlit server must receive the original Host header and websocket traffic. An nginx
location should include:

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

This configuration requires TLS and authentication to be configured by the proxy.

## Enrollment and recognition

1. Open the app through its HTTPS URL.
2. Use `Enroll student` and enter a valid student ID.
3. Upload several clear images containing exactly one face.
4. Select `Create or update templates`.
5. Templates are encrypted and stored in `/app/data/templates.enc`.
6. Use `Recognize faces` to upload an image or capture a browser image.

The web app enforces the quality thresholds and review policy from `config.yaml`. Browser
camera access requires HTTPS except for localhost development.

## Configuration

`config.yaml` controls model paths and recognition thresholds:

- `recognition.initial_threshold`: fallback cosine threshold
- `recognition.review_margin`: borderline-review margin
- `recognition.minimum_identity_margin`: ambiguity-review margin
- `paths.templates`: encrypted template gallery
- `detection.*`: detector, face-size, blur, and confidence thresholds

Deploy threshold changes only after independent validation. If `reports/threshold.json`
exists, `load_threshold()` uses its validation-selected threshold instead of the initial
configured value.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m attendance.download_models
$env:ATTENDANCE_TEMPLATE_KEY = python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
streamlit run app.py
```

For a shell that shows a literal key instead of an assignment, generate the key once and
set it through the environment variables dialog or your shell-specific export command.

## Validation checklist

Before production use:

1. Build the image successfully:
   `docker build --tag classroom-face-attendance:validation .`
2. Start it with a temporary test key and persistent volume.
3. Confirm `/_stcore/health` returns HTTP 200.
4. Confirm both ONNX files load and enrollment rejects unreadable/low-quality images.
5. Confirm recognition returns `Unknown` or `Review` as configured instead of silently
   approving uncertain matches.
6. Verify HTTPS, authentication, access logs, backups, key rotation, retention, and deletion.
7. Do not store raw biometric source images longer than policy allows.

## Updates and backup

Before an application update, back up the volume containing `data/templates.enc`. Changing
the Fernet key or losing the old key makes existing templates unreadable. Back up the key
through a secret manager, not beside the template file or in source control.
