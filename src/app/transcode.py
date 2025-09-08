import subprocess
from loguru import logger


def run(cmd: list[str]) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def transcode_to_mp4_with_auto_deinterlace(input_path: str, output_path: str) -> None:
    # ffmpeg filtergraph to auto-detect interlacing and apply yadif if interlaced
    # Use bwdif as higher-quality alternative if available; yadif is fine too
    filtergraph = (
        "idet,split[a][b];"
        "[a]yadif=deint=interlaced:parity=auto:mode=1[aout];"
        "[b]fps=fps=30000/1001[bout];"
        "[aout][bout]mergeplanes=0"
    )

    # However, mergeplanes trick is brittle. Prefer selective filter via 'bwdif=mode=auto:parity=auto' directly.
    # Simpler: always run bwdif; it is low-cost and preserves progressive frames.
    vf = "bwdif=mode=auto:parity=auto"

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


