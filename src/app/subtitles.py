import subprocess
from loguru import logger


def run(cmd: list[str]) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def mux_subtitles_into_mp4(input_mp4: str, input_srt: str, output_mp4: str) -> None:
    # Embed SRT as mov_text for broad Twitch compatibility.
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", input_mp4,
        "-i", input_srt,
        "-map", "0:v", "-map", "0:a?", "-map", "1:0",
        "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        output_mp4,
    ]
    run(cmd)


