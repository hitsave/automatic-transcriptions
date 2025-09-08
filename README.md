## Automatic Transcriptions Pipeline

End-to-end pipeline to fetch DVD/CD dumps from Wasabi S3, extract video, transcode to high-quality MP4 with automatic deinterlacing, transcribe and translate to English using WhisperX, and embed English subtitles for Twitch-ready streaming output.

### Requirements
- Docker and docker-compose
- Wasabi S3 credentials (via environment variables or AWS profile)

### Environment
Set your environment variables directly in `docker-compose.yml` or use the example file as a template.

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
docker compose up
```

Or run a one-off job:
```
docker compose run --rm worker
```

Data directories are mounted at `./data` on the host.

### Notes
- Transcoding uses libx264 CRF 18, preset slow, AAC 192k, faststart.
- Deinterlacing uses `bwdif` with automatic parity detection; progressive sources pass through.
- Subtitles are embedded as `mov_text` with `language=eng` and `default` disposition.
- For ISO/VIDEO_TS inputs, we attempt title 1. For multi-title DVDs, enhance `extract_main_title_to_mkv` to choose longest program.

