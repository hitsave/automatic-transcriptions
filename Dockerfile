FROM python:3.11-slim

# System deps: ffmpeg for transcode, p7zip for ISO extraction
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       p7zip-full \
       ca-certificates \
       tzdata \
       git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create directories for data volumes
RUN mkdir -p /data/incoming /data/work /data/output

COPY src/ /app/src/

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "src.app.main"]


