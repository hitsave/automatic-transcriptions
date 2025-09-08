import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
from .s3_client import WasabiClient
from .video_extract import extract_main_title_to_mkv
from .transcode import transcode_to_mp4_with_auto_deinterlace
from .whispering import transcribe_and_translate
from .subtitles import mux_subtitles_into_mp4


def load_processed_files(processed_file: str) -> dict:
    """Load the processed files tracking JSON."""
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load processed files tracking: {}", e)
    return {}


def save_processed_files(processed_file: str, processed_files: dict) -> None:
    """Save the processed files tracking JSON."""
    try:
        with open(processed_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except IOError as e:
        logger.error("Could not save processed files tracking: {}", e)


def find_main_media_file(s3_client, bucket: str, folder: str) -> dict | None:
    """Find the main media file in a folder (ISO or VIDEO_TS.IFO)."""
    # List objects in this specific folder
    folder_objs = s3_client.list_objects(bucket, folder + "/")
    
    # Priority order: ISO files first, then VIDEO_TS.IFO
    for obj in folder_objs:
        key = obj["Key"]
        if key.lower().endswith(".iso"):
            return obj
        if key.lower().endswith("video_ts.ifo"):
            return obj
    
    # If no main file found, return None
    return None


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
    dry_run: bool = False,
) -> None:
    work_root = os.path.join(local_cache, "work")
    incoming_root = os.path.join(local_cache, "incoming")
    output_root = os.path.join(local_cache, "output")
    os.makedirs(work_root, exist_ok=True)
    os.makedirs(incoming_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    # Load processed files tracking
    processed_file = os.path.join(local_cache, "processed_files.json")
    processed_files = load_processed_files(processed_file)

    s3 = WasabiClient(region=region, profile=profile)

    logger.info("Listing objects in bucket={}/{}", src_bucket, prefix)
    objs = s3.list_objects(src_bucket, prefix)
    
    # Group objects by folder/prefix to process each DVD/CD as a unit
    folders_to_process = set()
    for obj in objs:
        key = obj["Key"]
        # Check if this is a media file or folder we should process
        if (key.lower().endswith((".iso", "/video_ts/", "video_ts/", "video_ts.ifo")) or
            key.lower().endswith("/") or  # Folder
            any(key.lower().endswith(ext) for ext in [".vob", ".ifo", ".bup"])):  # DVD files
            # Extract the folder/prefix for this object
            if "/" in key:
                folder = "/".join(key.split("/")[:-1]) if not key.endswith("/") else key.rstrip("/")
            else:
                folder = key
            folders_to_process.add(folder)
    
    logger.info("Found {} folders to process: {}", len(folders_to_process), list(folders_to_process))
    
    for folder in sorted(folders_to_process):
        # Find the main media file in this folder
        main_file = find_main_media_file(s3, src_bucket, folder)
        if not main_file:
            logger.warning("No main media file found in folder: {}", folder)
            continue
            
        key = main_file["Key"]

        # Check if already processed
        if key in processed_files:
            logger.info("Skipping {} (already processed)", key)
            continue

        logger.info("Processing {}", key)
        # Preserve folder structure to avoid filename collisions
        # e.g., "E3_2004/E3_2004.iso" -> "E3_2004/E3_2004.iso"
        item_dir = os.path.join(incoming_root, os.path.dirname(key)) if "/" in key else incoming_root
        os.makedirs(item_dir, exist_ok=True)

        # Check if file already exists locally and is complete
        filename = os.path.basename(key)
        local_path = os.path.join(item_dir, filename)
        
        if os.path.exists(local_path):
            # Verify file is complete by checking size matches S3 object
            local_size = os.path.getsize(local_path)
            remote_size = obj.get("Size", 0)
            if local_size == remote_size and remote_size > 0:
                logger.info("File {} already exists locally and is complete ({} bytes), skipping download", local_path, local_size)
            else:
                logger.info("File {} exists but size mismatch (local: {} bytes, remote: {} bytes), re-downloading", local_path, local_size, remote_size)
                local_path = s3.download_object(src_bucket, key, item_dir)
        else:
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
        
        if dry_run:
            logger.info("DRY RUN: Would upload {} to s3://{}/{}", final_mp4_path, dst_bucket, out_key)
        else:
            s3.upload_file(dst_bucket, final_mp4_path, out_key)
            logger.info("Uploaded {} to s3://{}/{}", final_mp4_path, dst_bucket, out_key)

        # Mark as processed (even in dry run for testing)
        processed_files[key] = {
            "processed_at": datetime.now().isoformat(),
            "output_key": out_key,
            "size": obj.get("Size", 0),
            "last_modified": obj.get("LastModified", "").isoformat() if obj.get("LastModified") else "",
            "dry_run": dry_run
        }
        save_processed_files(processed_file, processed_files)
        logger.info("Marked {} as processed{}", key, " (dry run)" if dry_run else "")

        try:
            shutil.rmtree(item_dir, ignore_errors=True)
        except Exception:
            pass


