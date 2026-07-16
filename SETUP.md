# Local setup (Windows)

## Quick start: KSL Words and ASL Spelling

Install Git and Python 3.10 or newer, then clone and serve the project:

```powershell
git clone https://github.com/VITNT/asl-cool-testing.git
Set-Location asl-cool-testing
python -m http.server 8000
```

Open <http://localhost:8000/index.html>, select a recognition mode, click
**Start Camera**, and allow camera access.

KSL Words and ASL Spelling run in the browser. Keep an internet connection
available so TensorFlow.js and MediaPipe can load from their CDNs. VS Code Live
Server can be used instead of `python -m http.server 8000`.

## Optional: ASL Words

ASL Words requires the local WLASL300 Python service.

```powershell
git clone --depth 1 https://github.com/dxli94/WLASL.git third_party/WLASL
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r asl_service\requirements.txt
```

Download the official WLASL300 checkpoint from
<https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view>,
locate the `.pt` file containing `nslt_300_005100`, and copy it to:

```text
asl_service/models/wlasl300.pt
```

Then open a second Windows Command Prompt (`cmd.exe`) in the project folder and
start the Python backend server:

```cmd
.\.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Confirm <http://127.0.0.1:8001/health> reports `"status":"ready"`.

## Troubleshooting

- Camera unavailable: allow camera permission and use `localhost` instead of
  opening `index.html` directly.
- `No module named uvicorn`: run the command with
  `.\.venv\Scripts\python.exe`.
- Stale files after pulling: hard-refresh with `Ctrl+Shift+R`.
