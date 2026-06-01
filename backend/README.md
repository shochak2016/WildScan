# backend — WildScan inference API

FastAPI service that serves the models in [`../Models/`](../Models/) to the
[`../frontend/`](../frontend/). The frontend (on Vercel) uploads a file; this API
runs the models and returns the prediction.

## Endpoints
| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | `{status}` |
| GET | `/models` | — | which models loaded |
| POST | `/classify/image` | multipart `file` | runs **every** image model, returns the **top-confidence** result (auto-routes plant/animal/fungi) |
| POST | `/classify/audio` | multipart `file` | runs the **bird** model |

Response shape: `{ top: {label, scientific_name, examples, confidence, category, model}, candidates: [...], routed_to }`.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000   # run from backend/
curl -F file=@call.wav  localhost:8000/classify/audio
curl -F file=@photo.jpg localhost:8000/classify/image
```

## Models & Git LFS  ⚠️
Weights are tracked in **Git LFS**. A model is loaded only if its weight file has
**real bytes** (not an LFS pointer); pointer files are skipped with a log line.
So on a host that hasn't run `git lfs pull`, those models stay disabled.

- **bird** (audio) — always available (real weights in repo).
- **fungi** (image) — activates once `git lfs pull` materializes `Models/fungi/best_model.pth`.
- animal/plant image models: not wired yet (no labels / no committed weights — see repo README).

The repo-root `Dockerfile` copies `Models/`; enable **Git LFS** on the host so the
checkout has real weight bytes before the build (see `../DEPLOY.md`).

## Deploy
See [`../DEPLOY.md`](../DEPLOY.md) — backend on Render/Railway (Docker), frontend on Vercel.
