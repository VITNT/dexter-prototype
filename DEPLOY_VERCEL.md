# Deploy Dexter with Vercel and Hugging Face

This is the production deployment guide. The public site stays on one Vercel
URL, while the memory-heavy WLASL300 service runs in a free Hugging Face Docker
Space.

```text
Vercel
  static frontend
  /api/config
  /api/translate
       |
       | browser uploads a 2.5-second WebM clip
       v
Hugging Face Docker Space
  FastAPI /health and /predict
  PyTorch WLASL300 inference
       |
       | authenticated server-side download
       v
Private Vercel Blob
  wlasl/wlasl300.pt
  wlasl/pytorch_i3d.py
  wlasl/wlasl_class_list.txt
```

Do not commit `.env.local`, `wlasl300.pt`, the downloaded WLASL repository, or
the Blob token. WLASL is restricted to academic/computational, non-commercial
use by its authors.

## 1. Push the application repository

Push this repository to GitHub. The root `Dockerfile` is the Hugging Face
backend image. `.dockerignore` deliberately excludes the local checkpoint and
the downloaded `third_party/WLASL` tree from every Docker build context.

## 2. Create private model storage

1. Create or open the frontend project in Vercel.
2. Open **Storage**, create a **Blob** store with **Private** access, and connect
   it to the project.
3. Upload these files using the exact pathnames below.

| Local file | Private Blob pathname |
|---|---|
| `asl_service/models/wlasl300.pt` | `wlasl/wlasl300.pt` |
| `third_party/WLASL/code/I3D/pytorch_i3d.py` | `wlasl/pytorch_i3d.py` |
| `third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt` | `wlasl/wlasl_class_list.txt` |

With the Vercel CLI, first link the project and pull its environment:

```bash
npx vercel@latest login
npx vercel@latest link
npx vercel@latest env pull .env.local
npx vercel@latest blob put asl_service/models/wlasl300.pt --pathname wlasl/wlasl300.pt --access private
npx vercel@latest blob put third_party/WLASL/code/I3D/pytorch_i3d.py --pathname wlasl/pytorch_i3d.py --access private
npx vercel@latest blob put third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt --pathname wlasl/wlasl_class_list.txt --access private
```

Copy the generated `BLOB_READ_WRITE_TOKEN` from the connected store. It will be
added to Hugging Face as a secret in the next section. Never place it in a
frontend variable or commit it.

## 3. Create the Hugging Face Docker Space

1. In Hugging Face, create a new **Space**.
2. Choose **Docker** as the SDK and **CPU Basic** as the hardware.
3. Public visibility is sufficient for an academic demo. The source code and
   API will be public, but the Blob token and checkpoint remain private.
4. Upload this repository to the Space. The YAML block at the top of `README.md`
   selects Docker and exposes port `7860`.

The Hugging Face CLI avoids Git-history conflicts with the Space's generated
first commit and respects the repository's root `.gitignore`:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf upload YOUR_USERNAME/YOUR_SPACE . . --repo-type=space
```

In the Space **Settings**, add this secret:

```text
BLOB_READ_WRITE_TOKEN=<token copied from Vercel Blob>
```

Add these public variables:

```text
DEXTER_ALLOWED_ORIGINS=https://YOUR_PROJECT.vercel.app
WLASL_RUNTIME_ROOT=/tmp/dexter-wlasl
WLASL_MAX_CONCURRENT_INFERENCES=1
WLASL_INFERENCE_QUEUE_TIMEOUT_SECONDS=30
```

For Vercel preview deployments, optionally add a narrowly scoped regex:

```text
DEXTER_ALLOWED_ORIGIN_REGEX=^https://YOUR_PROJECT(-[a-zA-Z0-9-]+)?\.vercel\.app$
```

The Space downloads the three private assets on its first health request after
a cold start. Free Space disk is ephemeral, so another download is expected
after the Space sleeps or restarts. The application serializes inference to one
request at a time to prevent concurrent PyTorch calls from exhausting memory.
Requests that wait more than 30 seconds for that slot receive HTTP 429 instead
of building an unbounded queue.

Verify the Space directly:

```text
https://YOUR_USERNAME-YOUR_SPACE.hf.space/health
```

Expected response:

```json
{
  "status": "ready",
  "checkpointFound": true,
  "labelsFound": true,
  "classes": 300
}
```

## 4. Deploy the frontend to Vercel

Import the GitHub repository into Vercel with **Framework Preset: Other** and
the repository root as the Root Directory. Keep the default build, install, and
output settings.

Add these Vercel environment variables to Production and Preview:

```text
GEMINI_API_KEY=<your Google AI Studio key>
GEMINI_MODEL=gemma-4-31b-it
WLASL_API_URL=https://YOUR_USERNAME-YOUR_SPACE.hf.space
```

Do not add `BLOB_READ_WRITE_TOKEN` to browser-visible code. The Vercel frontend
does not need it. `WLASL_API_URL` is public configuration and must not have a
trailing slash.

Deploy, then verify:

```text
https://YOUR_PROJECT.vercel.app/
https://YOUR_PROJECT.vercel.app/api/config
```

`/api/config` must return the Space URL:

```json
{"wlaslApiUrl":"https://YOUR_USERNAME-YOUR_SPACE.hf.space"}
```

## 5. Production verification

1. Open the Vercel URL over HTTPS and grant camera access.
2. Select **ASL Words**.
3. If the free Space was sleeping, allow it time to wake and select the mode
   again if the first health request times out.
4. Record one complete sign and confirm `/predict` returns a result.
5. Confirm **Interpret Context** works through `/api/translate`.

## Troubleshooting

- **Space build cannot find WLASL files:** the Docker image intentionally does
  not copy them. Confirm `BLOB_READ_WRITE_TOKEN` is a Space secret.
- **`/health` returns `missing_model`:** confirm all three private Blob
  pathnames exactly match the table and the token belongs to that store.
- **Browser reports the backend is waking or unavailable:** open the Space
  `/health` URL once, wait for it to become ready, then retry ASL Words.
- **CORS error:** make `DEXTER_ALLOWED_ORIGINS` exactly match the Vercel origin,
  without a trailing slash, then restart the Space.
- **Prediction queues under multiple users:** this is intentional on free CPU
  hardware. `WLASL_MAX_CONCURRENT_INFERENCES=1` prevents memory exhaustion.
- **Clip upload returns 413:** keep each recorded clip below the host request
  limit; the app itself defaults to 20 MB outside Vercel.
- **Interpret Context fails:** verify the Vercel `GEMINI_API_KEY` and redeploy.

The public link to share remains only the Vercel URL. Visitors do not need the
Hugging Face or Blob credentials.
