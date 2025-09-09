FROM python:3.11-slim

# System deps: ffmpeg for transcode, libdvd-pkg for DVD copy protection, tools for IFO parsing
RUN echo "deb http://deb.debian.org/debian trixie main contrib non-free" > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libdvd-pkg \
       tesseract-ocr \
       tesseract-ocr-jpn \
       tesseract-ocr-eng \
       ca-certificates \
       tzdata \
       libdvdread-dev \
       libdvdnav-dev \
       lsdvd \
       python3-opencv \
       python3-numpy \
       python3-pil \
       p7zip-full \
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
