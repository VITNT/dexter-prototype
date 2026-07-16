# Dexter setup and deployment

For the production, one-link Vercel workflow, use
[DEPLOY_VERCEL.md](DEPLOY_VERCEL.md). This document covers local development
and the optional external-container alternative.

Dexter has two runtime pieces:

1. **Browser application** — ASL spelling and CSL/KSL recognition run in the
   browser. Vercel also runs the small `/api/translate` and `/api/config`
   JavaScript functions.
2. **Optional ASL Words service** — WLASL300 video inference runs in the
   `asl_service` FastAPI application. It requires PyTorch, the official WLASL
   source, and a separately obtained checkpoint.

The public site remains usable without the Python service: ASL spelling,
CSL/KSL words, the dictionary, and browser translation UI still load. Only
ASL Words is unavailable.

## Prerequisites

- Git
- Python 3.11 or 3.12
- Node.js 20 or newer for `npx vercel`
- Internet access: TensorFlow.js, MediaPipe, fonts, and landmark models are
  loaded from their upstream CDNs
- A webcam and a browser that supports `getUserMedia`
- Docker only when packaging the optional ASL Words backend

## 1. Fresh clone

```powershell
git clone <your-repository-url>
cd asl-cool-testing
Copy-Item .env.example .env.local
```

Edit `.env.local`:

```dotenv
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemma-4-31b-it
WLASL_API_URL=http://127.0.0.1:8001
```

`GEMINI_API_KEY` enables **Interpret Context**. Never place it in browser code
or commit `.env.local`. `WLASL_API_URL` is public configuration, not a secret.

## 2. Install the optional ASL Words service

Skip this section if only ASL spelling and CSL/KSL are needed.

Create the Python environment on Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r asl_service\requirements.txt
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r asl_service/requirements.txt
```

Install the official WLASL repository at the exact expected location:

```powershell
git clone https://github.com/dxli94/WLASL.git third_party/WLASL
```

Then follow [asl_service/models/README.md](asl_service/models/README.md) to
obtain the official WLASL300 checkpoint. These files must exist:

```text
asl_service/models/wlasl300.pt
third_party/WLASL/code/I3D/pytorch_i3d.py
third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt
```

WLASL is limited to academic/computational, non-commercial use. Its source and
checkpoint are intentionally not committed or included in the Vercel bundle.

Start the backend on Windows:

```powershell
.venv\Scripts\python.exe -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

On macOS/Linux:

```bash
.venv/bin/python -m uvicorn asl_service.app:app --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001/health>. Continue only when it returns
`"status":"ready"`.

## 3. Run the complete site locally

In a second terminal, run the Vercel development server from the repository
root:

```powershell
npx.cmd vercel@latest dev --listen 8000
```

The first run may ask you to sign in and link or create a Vercel project. Open
<http://localhost:8000/index.html>.

For browser recognition without the two JavaScript API functions, a basic
static server is enough:

```powershell
py -m http.server 8000
```

With the basic server, ASL spelling and CSL/KSL work, and ASL Words still uses
`http://127.0.0.1:8001`; **Interpret Context** does not work because
`/api/translate` requires the Vercel development server.

## 4. Package the ASL Words backend

The Docker build deliberately uses the locally installed restricted WLASL
files. Build it only after the health check above succeeds:

```powershell
docker build -f asl_service/Dockerfile -t dexter-asl .
docker run --rm -p 8001:8001 `
  -e DEXTER_ALLOWED_ORIGINS=http://localhost:8000 `
  dexter-asl
```

Deploy that image to a container service capable of running PyTorch. CPU works
but can be slow; GPU compute gives substantially better response time. Configure:

```text
PORT=8001
DEXTER_ALLOWED_ORIGINS=https://your-project.vercel.app
MAX_UPLOAD_BYTES=20971520
```

For changing Vercel preview domains, the backend can additionally use:

```text
DEXTER_ALLOWED_ORIGIN_REGEX=^https://[a-zA-Z0-9-]+\.vercel\.app$
```

Use an exact production origin whenever possible. Add authentication and
host-level rate limits before sharing an expensive public GPU endpoint.

## 5. Deploy to Vercel

Use the separate production-only [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md) guide.
It deploys the browser app and Python inference Function under one Vercel
domain, with the restricted WLASL assets stored privately in Vercel Blob.

## 6. Verify

```powershell
node --check script.js
Get-ChildItem src\pipeline\*.test.mjs | ForEach-Object {
  node $_.FullName
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
.venv\Scripts\python.exe -m unittest discover -s asl_service\tests -p "test_*.py"
```

Manual checks:

- ASL Spelling starts the camera and recognizes static letters.
- CSL/KSL loads the 75-class browser model.
- `POST /api/translate` works when `GEMINI_API_KEY` is present.
- ASL Words is enabled only when the configured backend reports ready.
- The deployed page receives camera permission over HTTPS.

## Files that must stay in Git

- `index.html`, `learn.html`, `styles.css`, and browser scripts
- `api/*.js` and `api/index.py`
- `src/pipeline/*.mjs` runtime modules
- `asl-model.json`, `asl-model.weights.bin`, and `asl-labels.json`
- `models/csl/**` and `models/asl/wlasl/manifest.json`
- `images/**`, `logo.png`, `vercel.json`, and `.vercelignore`
- `asl_service` source, root/Python requirements, Dockerfile, and model-install README

Files intentionally excluded from Git include secrets, virtual environments,
the WLASL repository/checkpoint, external datasets, recordings, caches, and
deployment-local state.
