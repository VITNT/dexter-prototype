# Simple local setup (Windows + VS Code)

VS Code **Go Live** only starts the website. ASL Words also needs the local
Python WLASL300 service. Complete steps 1–5 once per computer; after that, use
the two commands under **Every time you run the project**.

## One-time setup

### 1. Install prerequisites

Install:

- Python 3.10 or newer
- Git
- VS Code with the Live Server extension

### 2. Open the project folder

In PowerShell:

```powershell
Set-Location C:\path\to\asl-cool-testing
```

Replace the path with the location of your clone.

### 3. Download the official WLASL source

The official source is not committed inside this project. Run:

```powershell
git clone --depth 1 https://github.com/dxli94/WLASL.git third_party/WLASL
```

Read `third_party/WLASL/start_kit/C-UDA-1.0.pdf` before using the dataset or
checkpoint. The WLASL project restricts its material to academic/computational,
non-commercial use.

### 4. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r asl_service\requirements.txt
```

Always use `.\.venv\Scripts\python.exe`; using plain `python` afterward may
produce `No module named uvicorn` because it selects a different installation.

### 5. Install the WLASL300 checkpoint

1. Download the official `archived.zip`:
   <https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view>
2. Extract it.
3. Open `I3D/archived/asl300/` inside the extracted archive.
4. Prefer the `.pt` checkpoint containing `nslt_300_005100` in its name.
5. Copy it to `asl_service/models/`.
6. Rename the copied file to `wlasl300.pt`.

The final path must be:

```text
asl_service/models/wlasl300.pt
```

The model is deliberately excluded from Git.

## Every time you run the project

### 1. Start the ASL backend

Open a VS Code terminal at the project root and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Leave this terminal running. Open <http://127.0.0.1:8001/health> and confirm:

```json
{"status":"ready"}
```

The response includes other fields; `status` is the important one.

### 2. Start the website

In VS Code, open `index.html` and click **Go Live**. Live Server may use port
5500, 5501, or another local port; all local development ports are supported.

Select **ASL Words**, start the camera, and keep the backend terminal open.

## Common problems

### `No module named uvicorn`

You used the system Python. Run the backend with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

### Health says `missing_model`

Confirm this exact file exists:

```text
asl_service/models/wlasl300.pt
```

Also confirm the official repository exists at `third_party/WLASL/`.

### The page says the WLASL service is unavailable

- Confirm the backend terminal is still running.
- Open <http://127.0.0.1:8001/health>.
- Hard-refresh the website with `Ctrl+Shift+R`.

