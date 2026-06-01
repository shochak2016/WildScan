# Deploying WildScan

Two pieces: a **frontend** on Vercel and a **model-serving backend** on a container
host. They must be separate — **Vercel cannot run the models** (PyTorch + the model
files blow past the 250 MB serverless limit, and there's no persistent process).

```
[ Vercel: frontend/ (React) ]  --HTTPS-->  [ Render/Railway: backend/ (FastAPI + models) ]
```

## 1. Backend → Render (or Railway), Docker

The models are in **Git LFS**, so the host must pull them.

1. New **Web Service** from this repo, environment = **Docker**.
   - Dockerfile path: `backend/Dockerfile`, build context: repo root.
   - **Enable Git LFS** for the repo in the host settings (so weights download as real bytes, not pointers).
2. Env vars:
   - `CORS_ORIGINS=https://<your-vercel-app>.vercel.app` (lock CORS to your frontend).
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
