# WildScan — single-container deploy (frontend + inference API).
# Lives at the repo ROOT so hosts find it with default settings (context = root).
# HF Spaces / Render / Railway all build this.
#
# Model weights are in Git LFS: enable "Git LFS" so the checkout has REAL weight
# bytes (the API skips any weight that's still an LFS pointer, so it boots either way).

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=""          # same-origin: the API is served from this very container
RUN npm run build            # -> /fe/dist

# ---- Stage 2: Python API that also serves the built frontend ----
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 user          # HF Spaces runs containers as non-root uid 1000

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY Models /app/Models
COPY --from=frontend /fe/dist /app/frontend_dist   # static UI served at /

USER user
ENV PORT=7860 CORS_ORIGINS=*
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
