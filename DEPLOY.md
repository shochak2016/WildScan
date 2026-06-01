# Deploying WildScan

Two pieces: a **frontend** on Vercel and a **model-serving backend** on a container
host. They must be separate — **Vercel cannot run the models** (PyTorch + the model
files blow past the 250 MB serverless limit, and there's no persistent process).

```
[ Vercel: frontend/ (React) ]  --HTTPS-->  [ Hugging Face Space: backend/ (FastAPI + models) ]
```

## 1. Backend → Hugging Face Spaces (free, 16 GB RAM)

1. [huggingface.co/new-space](https://huggingface.co/new-space) → **SDK: Docker → "Blank"** template. Name it e.g. `wildscan-api`.
2. Push this repo to the Space's git remote (the repo-root `Dockerfile` + `README.md`
   frontmatter `sdk: docker`, `app_port: 7860` drive the build):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/wildscan-api
   git push space repo-reorg:main      # weights go via HF's free Git LFS
   ```
3. Space → **Settings → Variables and secrets** → add:
   - `LLM_API_KEY` = your free [Groq](https://console.groq.com/keys) key (enables `/assess`).
   - `CORS_ORIGINS` = `https://<your-vercel-app>.vercel.app` (optional; defaults to `*`).
4. Wait for the build, then verify (Space URL is `https://<you>-wildscan-api.hf.space`):
   ```bash
   curl https://<you>-wildscan-api.hf.space/health    # {"status":"ok"}
   curl https://<you>-wildscan-api.hf.space/models      # which models loaded
   ```
   `bird` always loads; `fungi` loads if its LFS weights pushed as real bytes.

> Free Spaces **sleep after ~48 h idle**; the first request after sleep cold-starts
> (~10–30 s to load torch + models). Fine for a demo.

**Alternatives** (same Dockerfile): **Railway** / **Fly.io** (cheap, scale-to-zero),
or **Render** (default `./Dockerfile`, enable Git LFS, ≥2 GB instance — pricier).

## 2. Frontend → Vercel

1. Import the repo in Vercel. **Set Root Directory = `frontend`** (the app lives in a subfolder).
   Framework preset auto-detects **Vite** (`vercel.json` is included).
2. Env var: `VITE_API_URL = https://<you>-wildscan-api.hf.space`.
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
