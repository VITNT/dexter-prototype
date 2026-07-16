# Local setup (Windows)

## Quick start: KSL Words and ASL Spelling

Install Git and Python 3.10 or newer, then clone and serve the project:

```powershell
git clone https://github.com/VITNT/asl-cool-testing.git
Set-Location asl-cool-testing
python -m http.server 8000
```

Open <http://localhost:8000/index.html>, select **KSL Words** or
**ASL Spelling**, click **Start Camera**, and allow camera access.

The KSL models are included in the repository. No package installation or
Python recognition backend is required. Keep an internet connection available
so the browser can load TensorFlow.js and MediaPipe from their CDNs.

VS Code Live Server can be used instead of `python -m http.server 8000`.

## Optional: ASL Words

ASL Words uses the local OpenHands ST-GCN WLASL2000 service and requires its
Python dependencies plus Git LFS for the included checkpoint.

### 1. Install the service

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r asl_service\requirements.txt
```

The model consumes the pose and hand landmarks already extracted in the
browser; the official WLASL I3D source is not required.

### 2. Download the OpenHands checkpoint

Install Git LFS before cloning or pulling the repository:

```powershell
git lfs install
git lfs pull
```

The checkpoint should be present at
`asl_service/models/openhands_wlasl2000_stgcn.ckpt` and approximately 41 MiB.
See `asl_service/models/README.md` for its exact size and SHA-256 checksum.

### 3. Start the ASL service

In a second terminal, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Confirm <http://127.0.0.1:8001/health> reports `"status":"ready"`, then select
**ASL Words** in the website.

## Troubleshooting

- `No module named uvicorn`: use `.\.venv\Scripts\python.exe`, not system Python.
- `missing_model`: confirm
  `asl_service/models/openhands_wlasl2000_stgcn.ckpt` is a full Git LFS file,
  not a small pointer.
- Camera unavailable: allow camera permission and open the site through
  `localhost`, not directly as a local file.
- Stale files after pulling: hard-refresh with `Ctrl+Shift+R`.
