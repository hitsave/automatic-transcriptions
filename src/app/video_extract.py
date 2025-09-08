import os
import subprocess
from loguru import logger


def run(cmd: list[str]) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_main_title_to_mkv(source_path: str, output_mkv: str) -> None:
    os.makedirs(os.path.dirname(output_mkv), exist_ok=True)
    lower = source_path.lower()
    if lower.endswith(".iso"):
        # Extract main title from ISO using ffmpeg vob demuxing
        # This is a simple approach: mount-like read via 7z to a temp dir is also possible
        # but ffmpeg can read directly from ISO titles.
        # Use title 1 by default; in real DVDs, detecting longest program would be ideal.
        input_url = f"dvd://1:{source_path}"
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-probesize", "100M", "-analyzeduration", "100M",
            "-i", input_url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "copy",
            output_mkv,
        ]
        run(cmd)
    elif "video_ts" in lower:
        # Read from VIDEO_TS folder; pick title 1
        input_url = f"dvd://1:{os.path.dirname(source_path) if source_path.lower().endswith('video_ts.ifo') else source_path}"
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-probesize", "100M", "-analyzeduration", "100M",
            "-i", input_url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "copy",
            output_mkv,
        ]
        run(cmd)
    else:
        # Already a file with video
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", source_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "copy",
            output_mkv,
        ]
        run(cmd)


