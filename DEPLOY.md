# Deploying WildScan

Two pieces: a **frontend** on Vercel and a **model-serving backend** on a container
host. They must be separate — **Vercel cannot run the models** (PyTorch + the model
files blow past the 250 MB serverless limit, and there's no persistent process).

```
[ Vercel: frontend/ (React) ]  --HTTPS-->  [ Render/Railway: backend/ (FastAPI + models) ]
```

## 1. Backend → Render (or Railway), Docker

The models are in **Git LFS**, so the host must pull them.

1. New **Web Service** from this repo, **Language/Runtime = Docker**.
   - The `Dockerfile` is at the **repo root** — leave Dockerfile Path at its **default** (`./Dockerfile`). Do *not* set a custom path.
   - **Enable Git LFS** for the repo in the host settings (so weights download as real bytes, not pointers — this is how the models get materialized).
2. Env vars:
   - `CORS_ORIGINS=https://<your-vercel-app>.vercel.app` (lock CORS to your frontend).
   - `LLM_API_KEY=<your key>` — enables the `/assess` threat summary. Free key from
     [Groq](https://console.groq.com/keys). (Optional: `LLM_BASE_URL`, `LLM_MODEL` to
     use Gemini/OpenRouter/OpenAI instead of Groq's default.)
   - `PORT` is provided by the host; the container respects it.
3. Deploy. Verify:
   ```bash
   curl https://<your-backend>.onrender.com/health     # {"status":"ok"}
   curl https://<your-backend>.onrender.com/models      # which models loaded
   ```
   `bird` should always load; `fungi` loads if LFS pulled its weights.

> First request is slow (model load / cold start). Use a paid/always-on instance to avoid sleeping.

## 2. Frontend → Vercel

1. Import the repo in Vercel. **Set Root Directory = `frontend`** (the app lives in a subfolder).
   Framework preset auto-detects **Vite** (`vercel.json` is included).
2. Env var: `VITE_API_URL = https://<your-backend>.onrender.com`.
3. Deploy. The app uploads images/audio to `${VITE_API_URL}/classify/{image,audio}`.

## Behavior
- **Image:** auto-routed — the backend runs every loaded image model and returns the most confident result (no category picker).
- **Audio:** birds only (the bird genus ensemble).
- If the backend is unreachable or a model isn't loaded, the UI shows a clear error with the API URL.

## Local dev
```bash
# backend
cd backend && pip install -r requirements.txt && uvicorn app:app --port 8000
# frontend (separate shell)
cd frontend && npm install && echo "VITE_API_URL=http://localhost:8000" > .env.local && npm run dev
```
