FROM python:3.11-slim

# System deps: ffmpeg for transcode, p7zip for ISO extraction, libdvd-pkg for DVD copy protection
RUN echo "deb http://deb.debian.org/debian trixie main contrib non-free" > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       p7zip-full \
       libdvd-pkg \
       ca-certificates \
       tzdata \
    && dpkg-reconfigure libdvd-pkg \
    && apt-get dist-clean

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create directories for data volumes
RUN mkdir -p /data/incoming /data/work /data/output

COPY src/ /app/src/

ENTRYPOINT ["python", "-m", "src.app.main"]
