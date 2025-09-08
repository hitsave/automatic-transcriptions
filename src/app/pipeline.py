import os
import shutil
from loguru import logger
from .s3_client import WasabiClient
from .video_extract import extract_main_title_to_mkv
from .transcode import transcode_to_mp4_with_auto_deinterlace
from .whispering import transcribe_and_translate
from .subtitles import mux_subtitles_into_mp4


def run_pipeline(
    src_bucket: str,
    prefix: str,
    dst_bucket: str,
    output_prefix: str,
    region: str,
    profile: str | None,
    local_cache: str,
    language: str,
    model_size: str,
) -> None:
    work_root = os.path.join(local_cache, "work")
    incoming_root = os.path.join(local_cache, "incoming")
    output_root = os.path.join(local_cache, "output")
    os.makedirs(work_root, exist_ok=True)
    os.makedirs(incoming_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    s3 = WasabiClient(region=region, profile=profile)

    logger.info("Listing objects in bucket={}/{}", src_bucket, prefix)
    objs = s3.list_objects(src_bucket, prefix)
    for obj in objs:
        key = obj["Key"]
        if not key.lower().endswith((".iso", "/video_ts/", "video_ts/", "video_ts.ifo")):
            continue

        logger.info("Processing {}", key)
        item_dir = os.path.join(incoming_root, key.replace("/", "_"))
        os.makedirs(item_dir, exist_ok=True)

        local_path = s3.download_object(src_bucket, key, item_dir)

        mkv_path = os.path.join(work_root, os.path.basename(item_dir) + ".mkv")
        try:
            extract_main_title_to_mkv(local_path, mkv_path)
        except Exception as e:
            logger.exception("Extraction failed for {}: {}", key, e)
            continue

        mp4_path = os.path.join(work_root, os.path.basename(item_dir) + ".mp4")
        try:
            transcode_to_mp4_with_auto_deinterlace(mkv_path, mp4_path)
        except Exception as e:
            logger.exception("Transcode failed for {}: {}", key, e)
            continue

        srt_path = os.path.join(work_root, os.path.basename(item_dir) + ".en.srt")
        try:
            transcribe_and_translate(mp4_path, srt_path, language=language, model_size=model_size)
        except Exception as e:
            logger.exception("Transcription failed for {}: {}", key, e)
            continue

        final_mp4_path = os.path.join(output_root, os.path.basename(item_dir) + ".final.mp4")
        try:
            mux_subtitles_into_mp4(mp4_path, srt_path, final_mp4_path)
        except Exception as e:
            logger.exception("Muxing failed for {}: {}", key, e)
            continue

        out_key = os.path.join(output_prefix, os.path.basename(final_mp4_path)) if output_prefix else os.path.basename(final_mp4_path)
        s3.upload_file(dst_bucket, final_mp4_path, out_key)

        try:
            shutil.rmtree(item_dir, ignore_errors=True)
        except Exception:
            pass


