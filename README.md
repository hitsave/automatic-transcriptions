## Automatic Transcriptions Pipeline

End-to-end pipeline to fetch DVD/CD dumps from Wasabi S3, extract video, transcode to high-quality MP4 with automatic deinterlacing, transcribe and translate to English using WhisperX, and embed English subtitles for Twitch-ready streaming output.

### Requirements
- Docker and docker-compose
- Wasabi S3 credentials (via `.env` or AWS profile)

### Environment
Create a `.env` file with:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
WASABI_SRC_BUCKET=your-source-bucket
WASABI_DST_BUCKET=your-destination-bucket
WASABI_PREFIX=path/to/dumps/
WASABI_OUTPUT_PREFIX=processed/
```

### Build
```
docker compose build
```

### Compose files
To avoid committing real credentials or paths, use the example compose as a starting point:
```
cp docker-compose.example.yml docker-compose.yml
```
Then edit `docker-compose.yml` locally with your real environment values. The `.gitignore` excludes it from version control.

### Run
Process all items under the prefix from source bucket and upload final MP4s with embedded English subtitles to destination bucket:
```
docker compose run --rm worker process \
  --src-bucket "$WASABI_SRC_BUCKET" \
  --prefix "$WASABI_PREFIX" \
  --dst-bucket "$WASABI_DST_BUCKET" \
  --output-prefix "$WASABI_OUTPUT_PREFIX" \
  --region "$AWS_REGION" \
  --local-cache /data \
  --language auto \
  --model-size medium
```

Data directories are mounted at `./data` on the host.

### Notes
- Transcoding uses libx264 CRF 18, preset slow, AAC 192k, faststart.
- Deinterlacing uses `bwdif` with automatic parity detection; progressive sources pass through.
- Subtitles are embedded as `mov_text` with `language=eng` and `default` disposition.
- For ISO/VIDEO_TS inputs, we attempt title 1. For multi-title DVDs, enhance `extract_main_title_to_mkv` to choose longest program.

