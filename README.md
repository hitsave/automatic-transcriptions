## Automatic Transcriptions Pipeline

End-to-end pipeline to fetch DVD/CD dumps from Wasabi S3, extract video, transcode to high-quality MP4 with automatic deinterlacing, transcribe and translate to English using WhisperX, and embed English subtitles for Twitch-ready streaming output.

### Requirements
- Docker and docker-compose
- Wasabi S3 credentials (via environment variables or AWS profile)
- **For GPU acceleration**: NVIDIA GPU with CUDA support + nvidia-docker2

### Environment
Set your environment variables directly in `docker-compose.yml` or use the example file as a template.

### Build

**CPU-only setup (compatible with any machine):**
```
docker compose build
```

**GPU-accelerated setup (requires NVIDIA GPU):**
```
docker compose -f docker-compose.gpu.yml build
```

### Compose files
To avoid committing real credentials or paths, use the example compose as a starting point:
```
cp docker-compose.example.yml docker-compose.yml
```
Then edit `docker-compose.yml` locally with your real environment values. The `.gitignore` excludes it from version control.

### Run
Process all items under the prefix from source bucket and upload final MP4s with embedded English subtitles to destination bucket:

**CPU-only setup:**
```bash
# Normal run (processes and uploads)
docker compose up

# One-off job
docker compose run --rm worker

# Dry run (processes locally but doesn't upload)
docker compose run --rm worker --dry-run
```

**GPU-accelerated setup (10-20x faster encoding):**
```bash
# Normal run (processes and uploads)
docker compose -f docker-compose.gpu.yml up

# One-off job
docker compose -f docker-compose.gpu.yml run --rm worker

# Dry run (processes locally but doesn't upload)
docker compose -f docker-compose.gpu.yml run --rm worker --dry-run
```

**Folder structure support:**
- Processes folders like `E3_2004/`, `F_WAVE_026/`, etc.
- Finds main media files (`.iso` or `VIDEO_TS.IFO`) within each folder
- Preserves folder structure locally to avoid filename collisions
- Skips metadata files (`.txt`, `.zip`, etc.)

Data directories are mounted at `./data` on the host.

### Features
- **Resumable processing**: Tracks completed files to avoid reprocessing
- **Smart downloads**: Skips re-downloading if file exists and is complete
- **Folder structure support**: Handles organized S3 buckets with multiple DVD/CD folders
- **Dry run mode**: Test processing without uploading to S3
- **Automatic deinterlacing**: Uses `bwdif` with automatic parity detection
- **Multi-language support**: Auto-detects language and translates to English
- **Twitch-ready output**: MP4 with embedded English subtitles

### Technical Details
- **Transcoding**: 
  - **CPU**: libx264 CRF 18, preset slow, AAC 192k, faststart
  - **GPU**: h264_nvenc CRF 18, preset p7, AAC 192k, faststart (10-20x faster)
- **GPU Detection**: Automatically detects NVIDIA GPU and uses NVENC encoder
- **Deinterlacing**: `bwdif` with automatic parity detection; progressive sources pass through
- **Subtitles**: Embedded as `mov_text` with `language=eng` and `default` disposition
- **Processing**: Sequential (one file at a time) for memory efficiency
- **Tracking**: JSON file at `/data/processed_files.json` records completed work

