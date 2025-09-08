from typer import Typer, Option
from typing import Optional
from loguru import logger
from .pipeline import run_pipeline

app = Typer(help="Automatic DVD/CD transcription and subtitle embedding pipeline")


@app.command()
def process(
    src_bucket: str = Option(
        None, help="Wasabi source bucket for inputs", envvar="WASABI_SRC_BUCKET"
    ),
    prefix: str = Option(
        "", help="Prefix within source bucket to scan (ISO/VIDEO_TS)", envvar="WASABI_PREFIX"
    ),
    dst_bucket: str = Option(
        None, help="Wasabi destination bucket for outputs", envvar="WASABI_DST_BUCKET"
    ),
    output_prefix: str = Option(
        "", help="Output prefix in destination bucket", envvar="WASABI_OUTPUT_PREFIX"
    ),
    region: str = Option("us-east-1", help="Wasabi region", envvar="AWS_REGION"),
    profile: Optional[str] = Option(
        None, help="AWS profile for Wasabi credentials (optional)", envvar="AWS_PROFILE"
    ),
    local_cache: str = Option("/data", help="Local working directory root (/data)"),
    language: str = Option("auto", help="Source language or 'auto' for detection"),
    model_size: str = Option("medium", help="WhisperX model size"),
):
    if not src_bucket:
        logger.error("WASABI_SRC_BUCKET environment variable is required")
        raise SystemExit(1)
    if not dst_bucket:
        logger.error("WASABI_DST_BUCKET environment variable is required")
        raise SystemExit(1)
    
    logger.info("Starting pipeline")
    run_pipeline(
        src_bucket=src_bucket,
        prefix=prefix,
        dst_bucket=dst_bucket,
        output_prefix=output_prefix,
        region=region,
        profile=profile,
        local_cache=local_cache,
        language=language,
        model_size=model_size,
    )


if __name__ == "__main__":
    app()


