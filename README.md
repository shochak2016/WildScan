# WildScan

Repo for WildScan, part of DS3 at UCSD Spring Projects. WildScan is a mobile-friendly web application that helps hikers, campers, and outdoor explorers quickly identify nearby animals using a photo or sound, and instantly understand whether they are safe.

Upload an **image** or an **audio clip** → WildScan classifies it, then an LLM writes a short summary and a **threat level** (safe / caution / dangerous).

- **Image** — auto-routed: the backend runs every available image model and returns the most confident result (no need to pick plant / animal / fungi).
- **Audio** — bird genus (50 North American genera).
- **Threat assessment** — after classification, an LLM returns a 3-sentence summary + safety level, shown in the UI.

---

## System design

```
  ┌─────────────┐   image / audio (multipart)   ┌──────────────────────────────┐
  │  Frontend   │ ────────────────────────────▶ │  Backend  (FastAPI)           │
  │ React + Vite│                                │  POST /classify/image|audio   │
  │  (upload UI)│ ◀──────────────────────────── │  POST /assess        (LLM)    │
  └─────────────┘   {species, confidence,        │  GET  /health /models         │
                     summary, threat_level}      └───────────────┬──────────────┘
                                                                 │ loads
                                              ┌──────────────────▼───────────────┐
                                              │ Models/  (PyTorch weights + code) │   ─▶ LLM API
                                              │ bird · fungi · animal · …         │     (Groq/Gemini)
                                              └───────────────────────────────────┘
```

- **`frontend/`** — React + Vite single-page app: upload (image/audio), result card, threat-level theming.
- **`backend/`** — FastAPI. `/classify/image` runs all *loadable* image models and returns the top-confidence one; `/classify/audio` runs the bird ensemble; `/assess` calls an LLM for the summary + threat level. It loads only models whose weight files are real (skips Git-LFS pointers), so it boots anywhere.
- **`Models/`** — per-model weights + inference code (`bird`, `fungi`, `plant`, `animal_phylum`, `california_animal`, `amphibian`). Weights are tracked in **Git LFS**.
- **`Data/`** — datasets + EDA/training notebooks (mirrors `Models/`).

In **production** the whole app runs as **one container** (`Dockerfile`): a Node stage builds the frontend, then FastAPI serves the static UI at `/` *and* the API. Deployed on **Hugging Face Spaces** — see [`DEPLOY.md`](DEPLOY.md).

### Layout
| Path | What |
|------|------|
| `frontend/` | React + Vite app |
| `backend/` | FastAPI inference API (`app.py`) |
| `Models/<name>/` | model weights + inference code |
| `Data/<name>/` | datasets + notebooks |
| `Dockerfile` | single-container build (frontend + API) |
| `DEPLOY.md` | deployment guide |

---

## Run locally

Two services: start the **backend**, then the **frontend**.

### 1. Backend (Python 3.12)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# model weights are in Git LFS — fetch the real bytes:
git lfs install && git lfs pull

# optional: enable LLM threat summaries (free key: https://console.groq.com/keys)
export LLM_API_KEY=hf_or_groq_key_here

uvicorn app:app --app-dir backend --port 8000
```
Verify: `curl localhost:8000/health` → `{"status":"ok"}`, and `curl localhost:8000/models` to see which models loaded (`bird` should always be there).

### 2. Frontend (Node 18+)
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev          # opens http://localhost:5173
```

---

## How to use

**In the app:** open it, then drag-and-drop or pick a **photo** or **audio clip**. Image uploads auto-route to the best model; audio runs the bird classifier. You'll see the predicted species, confidence, and the LLM threat summary.

**Via the API directly:**
```bash
# audio → bird genus
curl -F "file=@bird.mp3"  localhost:8000/classify/audio

# image → auto-routed to the most confident model
curl -F "file=@photo.jpg" localhost:8000/classify/image

# LLM threat summary for a species (needs LLM_API_KEY)
curl -X POST localhost:8000/assess \
     -H 'Content-Type: application/json' \
     -d '{"name":"Northern Cardinal (Cardinalis)"}'
```

Response shape:
```json
{ "top": {"label": "...", "scientific_name": "...", "confidence": 0.81, "category": "Birds", "model": "bird"},
  "candidates": [ ... ],
  "routed_to": "Birds" }
```

---

## Deploy
See [`DEPLOY.md`](DEPLOY.md) — single container on Hugging Face Spaces (free), or Render/Railway/Fly.
