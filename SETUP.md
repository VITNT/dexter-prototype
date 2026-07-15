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

ASL Words uses the local WLASL300 Python service and requires additional setup.

### 1. Install the WLASL source

```powershell
git clone --depth 1 https://github.com/dxli94/WLASL.git third_party/WLASL
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r asl_service\requirements.txt
```

Read `third_party/WLASL/start_kit/C-UDA-1.0.pdf` before using its dataset or
checkpoint; WLASL restricts its material to academic/computational,
non-commercial use.

### 2. Install the WLASL300 checkpoint

1. Download the official `archived.zip` from
   <https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view>.
2. Extract it and locate the `.pt` checkpoint containing `nslt_300_005100`
   under `I3D/archived/asl300/`.
3. Copy it to `asl_service/models/wlasl300.pt`.

### 3. Start the ASL service

In a second terminal, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Confirm <http://127.0.0.1:8001/health> reports `"status":"ready"`, then select
**ASL Words** in the website.

## Troubleshooting

- `No module named uvicorn`: use `.\.venv\Scripts\python.exe`, not system Python.
- `missing_model`: confirm both `asl_service/models/wlasl300.pt` and
  `third_party/WLASL/` exist.
- Camera unavailable: allow camera permission and open the site through
  `localhost`, not directly as a local file.
- Stale files after pulling: hard-refresh with `Ctrl+Shift+R`.
