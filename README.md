# WildScan

Repo for WildScan, part of DS3 at UCSD Spring Projects. WildScan is a mobile-friendly web application that helps hikers, campers, and outdoor explorers quickly identify nearby wildlife using a photo or sound, and instantly understand whether they are safe.

Upload an **image** or an **audio clip** → WildScan classifies it, then an LLM writes a short summary and a **threat level** (safe / caution / dangerous).

- **Image** — auto-routed: the backend runs the available image models and returns the most confident result (no need to pick plant / animal / fungi).
- **Audio** — identifies wildlife by their calls.
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
                                              │ per-taxon classifiers             │     (Groq/Gemini)
                                              └───────────────────────────────────┘
```

- **`frontend/`** — React + Vite single-page app: upload (image/audio), result card, threat-level theming.
- **`backend/`** — FastAPI. `/classify/image` runs the image models and returns the top-confidence result; `/classify/audio` runs the audio model; `/assess` calls an LLM for the summary + threat level.
- **`Models/`** — one folder per classifier: weights + inference code. Weights are tracked in **Git LFS**.
- **`Data/`** — datasets + EDA / training notebooks (mirrors `Models/`).

In **production** the whole app runs as **one container** (`Dockerfile`): a Node stage builds the frontend, then FastAPI serves the static UI at `/` *and* the API. Deployed on **Hugging Face Spaces** — see [`DEPLOY.md`](DEPLOY.md).

---

## Repository structure

```
WildScan/
├── frontend/                 React + Vite single-page app (the UI)
│   ├── index.html
│   ├── package.json          deps + scripts (dev / build)
│   ├── vite.config.ts        build config (@ → src alias)
│   ├── postcss.config.mjs    Tailwind / PostCSS
│   ├── vercel.json           SPA config (if hosting the UI on Vercel)
│   ├── .env.example          VITE_API_URL (backend URL)
│   └── src/
│       ├── main.tsx          app entry
│       ├── app/
│       │   ├── App.tsx       upload → API call → result + threat theming
│       │   └── components/   UploadSection · ResultCard · ScanHistory
│       │                     · ThreatBadge · ScaleComparison
│       └── styles/           Tailwind + theme CSS
│
├── backend/                  FastAPI inference API
│   ├── app.py                endpoints (/classify, /assess, /health, /models),
│   │                         model registry, LLM threat assessment, static serving
│   ├── requirements.txt      API + ML deps (CPU PyTorch, librosa, …)
│   └── README.md             API reference
│
├── Models/                   trained weights + inference code — one folder per model
│   ├── bird/                 bird-sound classifier
│   │   ├── bird_classifier.py    pipeline: download → preprocess → train → ensemble → predict
│   │   ├── model_b0.pth          weights
│   │   ├── model_b1.pth          weights
│   │   ├── class_mapping.json    inference config (arch, norm, spectrogram params)
│   │   ├── label_vocabs.json     class labels + common names
│   │   ├── genus_list.json       taxa covered
│   │   ├── requirements.txt
│   │   └── README.md             model card (dataset, training, results)
│   ├── fungi/                fungi image classifier (predict_fungi.py + weights + labels)
│   ├── plant/                plant image model (train_model.py)
│   ├── animal_phylum/        animal phylum/class image model (weights)
│   ├── california_animal/    regional animal species image model (weights)
│   └── amphibian/            amphibian-sound classifier (deployment bundle)
│
├── Data/                     datasets + EDA / training notebooks (mirrors Models/)
│   ├── bird/                 eda.ipynb (dataset exploration)
│   ├── fungi/                training notebook + dataframe
│   ├── plant/                labels.csv, observations.csv
│   ├── animal_phylum/        data-processing notebook
│   ├── california_animal/    model + EDA notebooks, image catalog
│   ├── amphibian/            sound dataset CSVs + labeling script
│   └── wildscan_data/        shared image dataset catalog
│
├── Dockerfile                single-container build: Node builds UI → FastAPI serves UI + API
├── DEPLOY.md                 deployment guide (Hugging Face Spaces / Render / Railway)
├── README.md                 this file
└── LICENSE
```

**Where things live:** the **UI** is entirely in `frontend/`; the **server + routing/LLM logic** is `backend/app.py`; each **model** is a self-contained folder in `Models/<name>/` (weights + the code to run it); and the **data + notebooks** that produced each model sit in the matching `Data/<name>/`.

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
export LLM_API_KEY=your_key_here

uvicorn app:app --app-dir backend --port 8000
```
Verify: `curl localhost:8000/health` → `{"status":"ok"}`; `curl localhost:8000/models` lists the loaded models.

### 2. Frontend (Node 18+)
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev          # opens http://localhost:5173
```

---

## How to use

**In the app:** open it, then drag-and-drop or pick a **photo** or **audio clip**. The image path auto-routes to the best-matching model; audio is classified by the sound model. You'll see the predicted species, confidence, and the LLM threat summary.

**Via the API directly:**
```bash
# image → auto-routed to the most confident model
curl -F "file=@photo.jpg" localhost:8000/classify/image

# audio clip
curl -F "file=@clip.wav"  localhost:8000/classify/audio

# LLM threat summary for a species (needs LLM_API_KEY)
curl -X POST localhost:8000/assess \
     -H 'Content-Type: application/json' \
     -d '{"name":"Northern Cardinal"}'
```

Response shape:
```json
{ "top": {"label": "...", "scientific_name": "...", "confidence": 0.81, "category": "...", "model": "..."},
  "candidates": [ ... ],
  "routed_to": "..." }
```

---

## Deploy
See [`DEPLOY.md`](DEPLOY.md) — single container on Hugging Face Spaces (free), or Render / Railway / Fly.
