"""
WildScan inference API (FastAPI).

Serves the trained models in ../Models over HTTP for the Vercel frontend.

Routing
-------
- POST /classify/image  (multipart 'file'): runs EVERY loadable image model and
  returns the single highest-confidence prediction (auto-routes plant/animal/
  fungi without the user choosing a category).
- POST /classify/audio  (multipart 'file'): runs the bird model.
- GET  /health, GET /models: status + which models actually loaded.

Weights live in Git LFS. A model is loaded only if its weight file has REAL
bytes (not an LFS pointer); pointer files are skipped with a warning, so this
runs anywhere and lights up more models once `git lfs pull` has run (see
Dockerfile).

Run locally:  uvicorn app:app --reload --port 8000
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent          # repo root
MODELS = ROOT / "Models"

# LLM for threat assessment — OpenAI-compatible (Groq by default; works with
# Gemini/OpenRouter/OpenAI by changing LLM_BASE_URL + LLM_MODEL). Key from env.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="WildScan Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_real_weights(path: Path) -> bool:
    """True if `path` exists and is NOT a Git-LFS pointer stub."""
    if not path.exists():
        return False
    if path.stat().st_size > 4096:
        return True
    head = path.read_bytes()[:64]
    return not head.startswith(b"version https://git-lfs")


def _save_tmp(upload: UploadFile, suffix: str) -> str:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp


# ---------------------------------------------------------------------------
# Model wrappers — each exposes .name/.category and .predict(path)->list[dict]
# A prediction dict: {label, scientific_name, confidence, category, model}
# ---------------------------------------------------------------------------
class BirdModel:
    name, category, kind = "bird", "Birds", "audio"

    def __init__(self):
        sys.path.insert(0, str(MODELS / "bird"))
        import torch
        import bird_classifier as bc
        self.bc, self.torch = bc, torch
        self.models, self.cfg, self.vocabs = bc.load_models(str(MODELS / "bird"), torch.device("cpu"))

    def predict(self, path, top_k=5):
        preds, _ = self.bc.predict(path, self.models, self.cfg, self.vocabs,
                                   self.torch.device("cpu"), top_k=top_k)
        return [{"label": p["common_label"], "scientific_name": p["genus"],
                 "examples": p.get("examples", ""), "confidence": p["confidence"],
                 "category": self.category, "model": self.name} for p in preds]


class FungiModel:
    name, category, kind = "fungi", "Fungi", "image"

    def __init__(self):
        sys.path.insert(0, str(MODELS / "fungi"))
        import torch
        import predict_fungi as pf
        self.pf, self.torch = pf, torch
        self.model, self.vocabs = pf.load_model(
            str(MODELS / "fungi" / "best_model.pth"),
            str(MODELS / "fungi" / "label_vocabs.json"), torch.device("cpu"))

    def predict(self, path, top_k=5):
        r = self.pf.predict(path, self.model, self.vocabs, self.torch.device("cpu"), top_k=top_k)
        out = []
        for p in r["class"][:top_k]:                       # class-level is the reliable head
            out.append({"label": p["common_name"], "scientific_name": p["scientific_name"],
                        "examples": p.get("examples", ""), "confidence": p["confidence"],
                        "category": self.category, "model": self.name})
        return out


class AnimalModel:
    name, category, kind = "animal", "Animals", "image"
    WEIGHTS = "california_animal/final_production_wildscan_model.pth"

    def __init__(self):
        import torch
        import timm
        from torchvision import transforms
        self.torch = torch
        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=30)
        sd = torch.load(MODELS / self.WEIGHTS, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.model.load_state_dict(sd)
        self.model.eval()
        # matches training: 256px, ImageNet normalization
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # recovered label mapping (sorted species = training label order)
        lp = MODELS / "california_animal" / "labels.json"
        if lp.exists():
            d = json.load(open(lp))
            self.species, self.common = d["species"], d.get("common", {})
        else:
            self.species, self.common = None, {}

    def predict(self, path, top_k=5):
        import torch.nn.functional as F
        from PIL import Image
        x = self.tf(Image.open(path).convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0].numpy()
        order = probs.argsort()[::-1][:top_k]
        out = []
        for i in order:
            i = int(i)
            sci = self.species[i] if self.species else ""
            label = self.common.get(sci, sci) if sci else f"Animal (class {i})"
            out.append({"label": label, "scientific_name": sci, "examples": "",
                        "confidence": float(probs[i]), "category": self.category, "model": self.name})
        return out


# Register model classes with the weight file that gates loading them.
IMAGE_MODELS = [
    (FungiModel, MODELS / "fungi" / "best_model.pth"),
    (AnimalModel, MODELS / "california_animal" / "final_production_wildscan_model.pth"),
]
AUDIO_MODELS = [(BirdModel, MODELS / "bird" / "model_b0.pth")]

LOADED = {"image": [], "audio": []}


@app.on_event("startup")
def _load():
    for kind, registry in (("image", IMAGE_MODELS), ("audio", AUDIO_MODELS)):
        for cls, weight in registry:
            if not has_real_weights(weight):
                print(f"[skip] {cls.name}: no real weights at {weight} (LFS pointer?)")
                continue
            try:
                LOADED[kind].append(cls())
                print(f"[ok]   loaded {cls.name} ({kind})")
            except Exception as e:
                print(f"[fail] {cls.name}: {e}")
    print(f"Loaded image={[m.name for m in LOADED['image']]} audio={[m.name for m in LOADED['audio']]}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"image": [m.name for m in LOADED["image"]],
            "audio": [m.name for m in LOADED["audio"]]}


@app.post("/classify/image")
async def classify_image(file: UploadFile = File(...)):
    if not LOADED["image"]:
        raise HTTPException(503, "No image models available on this host.")
    tmp = _save_tmp(file, Path(file.filename or "img").suffix or ".jpg")
    try:
        # Run EVERY image model, then auto-route to the most confident result.
        all_preds = []
        for m in LOADED["image"]:
            try:
                all_preds += m.predict(tmp)
            except Exception as e:
                print(f"[warn] {m.name} failed: {e}")
        if not all_preds:
            raise HTTPException(500, "All image models failed")
        all_preds.sort(key=lambda p: p["confidence"], reverse=True)
        return {"top": all_preds[0], "candidates": all_preds[:5],
                "routed_to": all_preds[0]["category"]}
    finally:
        os.unlink(tmp)


@app.post("/classify/audio")
async def classify_audio(file: UploadFile = File(...)):
    if not LOADED["audio"]:
        raise HTTPException(503, "No audio models available on this host.")
    tmp = _save_tmp(file, Path(file.filename or "aud").suffix or ".wav")
    try:
        preds = LOADED["audio"][0].predict(tmp)
        return {"top": preds[0], "candidates": preds[:5], "routed_to": preds[0]["category"]}
    finally:
        os.unlink(tmp)


@app.post("/assess")
async def assess(payload: dict):
    """LLM threat assessment for a classified species. Body: {"name": "..."}.
    Returns a 3-sentence summary + a threat level (safe|caution|dangerous).
    Key is server-side (LLM_API_KEY) so it never reaches the browser."""
    name = (payload.get("name") or payload.get("label") or "").strip()
    if not name:
        raise HTTPException(400, "Missing 'name'")
    if not LLM_API_KEY:
        raise HTTPException(503, "Threat assessment not configured (set LLM_API_KEY).")

    system = ('You are a wildlife safety assistant. Respond ONLY with JSON: '
              '{"summary": "<exactly 3 sentences>", "threat_level": "safe|caution|dangerous"}. '
              'threat_level reflects danger to humans.')
    user = f"give me a 3 sentence summary on {name} and a threat level to humans"
    body = {"model": LLM_MODEL, "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    try:
        r = requests.post(f"{LLM_BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                          json=body, timeout=30)
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error {r.status_code}: {r.text[:200]}")

    content = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        summary = data.get("summary", content)
        level = str(data.get("threat_level", "caution")).lower()
    except Exception:
        summary, level = content, "caution"
    if level not in ("safe", "caution", "dangerous"):
        level = "caution"
    return {"summary": summary, "threat_level": level}


# ---------------------------------------------------------------------------
# Serve the built React frontend at "/" (the API routes above take precedence).
# Present only when built into the image (see Dockerfile stage 1).
# ---------------------------------------------------------------------------
_FE = ROOT / "frontend_dist"
if _FE.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FE), html=True), name="frontend")
