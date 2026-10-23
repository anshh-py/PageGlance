FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/pyproject.toml ./
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./

RUN pip install .

EXPOSE 10000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
