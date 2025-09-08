import subprocess
from loguru import logger


def run(cmd: list[str]) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def transcode_to_mp4_with_auto_deinterlace(input_path: str, output_path: str) -> None:
    # Use bwdif for deinterlacing - it's low-cost and preserves progressive frames
    # Fix the syntax: mode should be 0, 1, or 2, not "auto"
    vf = "bwdif=mode=1:parity=auto"

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-vf", vf,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    run(cmd)


