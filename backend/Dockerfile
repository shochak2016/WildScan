# WildScan inference API — deploy on Render/Railway (CPU container).
# Build context = repo root:  docker build -f backend/Dockerfile -t wildscan-api .
FROM python:3.12-slim

# git-lfs so model weights (tracked in LFS) can be materialized at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the API + the models it serves
COPY backend /app/backend
COPY Models /app/Models

# If the .git dir is present and weights are still LFS pointers, pull them.
# (Render: enable Git LFS for the repo, or the checkout already has real bytes.)
COPY .git /app/.git
RUN git lfs install || true && \
    (cd /app && git lfs pull || echo "git lfs pull skipped/failed — ensure weights are real") && \
    rm -rf /app/.git

ENV PORT=8000 CORS_ORIGINS=*
EXPOSE 8000
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
