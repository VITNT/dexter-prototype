# Deploy Dexter from GitHub to one Vercel URL

This is the production guide. Visitors only open the final Vercel URL and
grant camera permission. They do not install Python, models, or dependencies.

The deployment contains:

- the static browser application;
- ASL spelling and CSL/KSL recognition in the browser;
- `/api/translate` for context translation;
- `/api/asl/*` as a Vercel Python Function for ASL Words;
- a private Vercel Blob store containing the three restricted WLASL files.

## 1. Push the repository to GitHub

Make sure the files listed in [SETUP.md](SETUP.md#files-that-must-stay-in-git)
are committed and pushed. Do **not** commit `.env.local`, `wlasl300.pt`, or the
downloaded `third_party/WLASL` repository.

## 2. Import the GitHub repository

1. Open <https://vercel.com/new>.
2. Find the GitHub repository and select **Import**.
3. Set **Framework Preset** to **Other**.
4. Keep **Root Directory** as `./`.
5. Leave Build Command, Output Directory, and Install Command at their
   defaults. The repository's `vercel.json` is already configured.
6. Before deploying, add these Environment Variables for Production, Preview,
   and Development:

```text
GEMINI_API_KEY=<your Google AI Studio key>
GEMINI_MODEL=gemma-4-31b-it
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

Do not add `WLASL_API_URL`. When it is absent, the browser automatically uses
the same Vercel domain at `/api/asl`.

7. Select **Deploy**.

The first deployment can finish before the WLASL blobs are uploaded. ASL
spelling and CSL/KSL should already work; ASL Words will report a missing model
until the next section is complete.

## 3. Create private model storage

1. Open the deployed project in the Vercel dashboard.
2. Open **Storage**.
3. Select **Create Database**, then **Blob**.
4. Choose **Private** access and connect the store to this project. Vercel adds
   the required `BLOB_READ_WRITE_TOKEN` automatically.
5. Upload the following local files using these exact Blob pathnames:

| Local file | Private Blob pathname |
|---|---|
| `asl_service/models/wlasl300.pt` | `wlasl/wlasl300.pt` |
| `third_party/WLASL/code/I3D/pytorch_i3d.py` | `wlasl/pytorch_i3d.py` |
| `third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt` | `wlasl/wlasl_class_list.txt` |

If the dashboard does not offer file upload, install/login to the Vercel CLI,
pull the project environment, and upload with private access:

```powershell
npx.cmd vercel@latest login
npx.cmd vercel@latest link
npx.cmd vercel@latest env pull .env.local
npx.cmd vercel@latest blob put asl_service/models/wlasl300.pt --pathname wlasl/wlasl300.pt --access private
npx.cmd vercel@latest blob put third_party/WLASL/code/I3D/pytorch_i3d.py --pathname wlasl/pytorch_i3d.py --access private
npx.cmd vercel@latest blob put third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt --pathname wlasl/wlasl_class_list.txt --access private
```

The exact Blob CLI flags can change; run
`npx.cmd vercel@latest blob put --help` if `--pathname` is rejected.

## 4. Redeploy and verify

After connecting the Blob store, open **Deployments**, select the latest
deployment's menu, and choose **Redeploy** so the Function receives the Blob
credentials.

Verify these URLs, replacing the domain:

```text
https://your-project.vercel.app/
https://your-project.vercel.app/api/config
https://your-project.vercel.app/api/asl/health
```

Expected health response:

```json
{
  "status": "ready",
  "checkpointFound": true,
  "labelsFound": true,
  "classes": 300
}
```

The first ASL health check or prediction after a cold start downloads the
50 MB checkpoint into the Function's temporary disk and loads PyTorch, so it
can be noticeably slower than later requests. Vercel Hobby Functions have
2 GB memory and one managed vCPU; monitor **Project**, **Observability**, then
**Functions** for timeouts or memory failures.

## 5. Share one link

Share only:

```text
https://your-project.vercel.app/
```

GitHub pushes to the production branch trigger new production deployments.
Other branches and pull requests receive Preview URLs. You can add a custom
domain later under **Project**, **Settings**, then **Domains**.

## Troubleshooting

- **Build exceeds the standard Function size:** confirm Fluid Compute is on
  and `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` is set, then redeploy.
- **`/api/asl/health` says `missing_model`:** confirm all three private Blob
  pathnames match exactly and the Blob store is connected to this project.
- **ASL clip returns 413:** Vercel limits Function request bodies to 4.5 MB.
  The app's normal 2.5-second WebM clips should remain below that limit; check
  browser recording bitrate if they do not.
- **Camera is blocked:** use the HTTPS Vercel URL and allow camera permission.
- **Interpret Context fails:** verify `GEMINI_API_KEY`, then redeploy.
- **Inference is too slow on Hobby:** this I3D model is CPU-heavy. At that
  point, move only `/api/asl` to a GPU/container host and set `WLASL_API_URL`;
  visitors still keep using the same frontend link.
