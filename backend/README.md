# backend — WildScan inference API

FastAPI service that serves the models in [`../Models/`](../Models/) to the
[`../frontend/`](../frontend/). The frontend uploads a file; this API runs the
models and returns the prediction (plus an optional LLM threat summary).

## Endpoints
| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | `{status}` |
| GET | `/models` | — | which models are loaded |
| POST | `/classify/image` | multipart `file` | runs the image models, returns the **top-confidence** result (auto-routes plant/animal/fungi) |
| POST | `/classify/audio` | multipart `file` | runs the audio model |
| POST | `/assess` | `{"name": "..."}` | LLM 3-sentence summary + threat level (needs `LLM_API_KEY`) |

Response shape: `{ top: {label, scientific_name, examples, confidence, category, model}, candidates: [...], routed_to }`.

## Run locally
```bash
pip install -r requirements.txt
git lfs install && git lfs pull          # fetch model weights (Git LFS)
uvicorn app:app --reload --port 8000     # run from backend/
curl -F file=@photo.jpg localhost:8000/classify/image
curl -F file=@clip.wav  localhost:8000/classify/audio
```

## Environment
| Var | Purpose |
|-----|---------|
| `LLM_API_KEY` | enables `/assess` (free key: [Groq](https://console.groq.com/keys)) |
| `LLM_BASE_URL`, `LLM_MODEL` | optional — point at Gemini / OpenRouter / OpenAI instead of Groq |
| `CORS_ORIGINS` | comma-separated allowed origins (default `*`) |
| `PORT` | server port (default 7860) |

## Models & Git LFS
Model weights are tracked in **Git LFS**, loaded at startup from `../Models/`. Enable
Git LFS on the host (or run `git lfs pull`) so the checkout has the real weight bytes.
The API loads whichever models are present and logs the active set at startup
(`GET /models`).

## Deploy
See [`../DEPLOY.md`](../DEPLOY.md) — single container on Hugging Face Spaces (or Render / Railway).
