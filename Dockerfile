# WildScan — single-container deploy (frontend + inference API).
# Lives at the repo ROOT so hosts build it with default settings (context = root).
# HF Spaces / Render / Railway all build this.
# Model weights are in Git LFS; enable LFS so the checkout has real weight bytes.

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# same-origin: the API is served from this very container
ENV VITE_API_URL=""
RUN npm run build

# ---- Stage 2: Python API that also serves the built frontend ----
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 user

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY Models /app/Models
# static UI (built in stage 1), served at /
COPY --from=frontend /fe/dist /app/frontend_dist

USER user
ENV PORT=7860
ENV CORS_ORIGINS=*
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
