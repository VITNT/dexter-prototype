import assert from "node:assert/strict";
import fs from "node:fs";

const root = new URL("../../", import.meta.url);
const dockerfile = fs.readFileSync(new URL("Dockerfile", root), "utf8");
const dockerignore = fs.readFileSync(new URL(".dockerignore", root), "utf8");
const vercelignore = fs.readFileSync(new URL(".vercelignore", root), "utf8");
const vercelConfig = JSON.parse(fs.readFileSync(new URL("vercel.json", root), "utf8"));
const backendRequirements = fs.readFileSync(
  new URL("asl_service/requirements.txt", root),
  "utf8"
);
const backendApp = fs.readFileSync(new URL("asl_service/app.py", root), "utf8");
const backendAssets = fs.readFileSync(new URL("asl_service/assets.py", root), "utf8");

assert.equal(
  fs.existsSync(new URL("api/index.py", root)),
  false,
  "Vercel must not package the PyTorch FastAPI backend."
);
assert.deepEqual(
  Object.keys(vercelConfig.functions),
  ["api/*.js"],
  "Only the lightweight JavaScript routes belong in the Vercel function bundle."
);
assert.match(vercelignore, /^asl_service\/$/m);
assert.match(vercelignore, /^Dockerfile$/m);

assert.match(dockerfile, /FROM python:3\.12-slim/);
assert.match(dockerfile, /EXPOSE 7860/);
assert.match(dockerfile, /WLASL_RUNTIME_ROOT=\/tmp\/dexter-wlasl/);
assert.match(dockerfile, /--port \$\{PORT:-7860\}/);
assert.doesNotMatch(
  dockerfile,
  /COPY\s+third_party|COPY\s+.*wlasl300\.pt/,
  "Restricted WLASL assets must never be copied into the public Space image."
);
assert.doesNotMatch(
  dockerignore,
  /!third_party|!.*\.pt/,
  "Restricted local assets must stay outside the Docker build context."
);

assert.match(backendRequirements, /torch==2\.7\.1\+cpu/);
assert.match(backendRequirements, /vercel>=0\.5\.0,<1/);
assert.match(backendAssets, /BLOB_READ_WRITE_TOKEN/);
assert.match(backendApp, /asyncio\.Semaphore/);
assert.match(backendApp, /inference_semaphore\.acquire/);
assert.match(backendApp, /status_code=429/);

console.log("Split deployment contract passed: Vercel frontend, Hugging Face backend, private Blob assets.");
