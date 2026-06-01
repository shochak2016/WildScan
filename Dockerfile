# WildScan inference API — deploy on Render/Railway (CPU container).
# Lives at the repo ROOT so hosts find it with default settings (context = root).
#
# Model weights are in Git LFS: enable "Git LFS" for the repo on your host so the
# checkout has REAL weight bytes before this build runs (the API skips any weight
# that's still an LFS pointer, so it still boots either way).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 user          # HF Spaces runs containers as non-root uid 1000

WORKDIR /app

# Python deps first (layer cache)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# API + models (world-readable; the non-root user can read them)
COPY backend /app/backend
COPY Models /app/Models

USER user
# HF Spaces expects the app on app_port (7860 here). Render/Railway set $PORT and override.
ENV PORT=7860 CORS_ORIGINS=*
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
