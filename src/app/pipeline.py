import os
import shutil
import json
import hashlib
from datetime import datetime
from pathlib import Path
from loguru import logger
from .s3_client import WasabiClient
from .video_extract import extract_main_title_to_mp4
from .transcode import transcode_to_mp4_with_auto_deinterlace
# from .whispering import transcribe_and_translate  # Temporarily disabled
# from .subtitles import mux_subtitles_into_mp4  # Temporarily disabled


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


def verify_file_integrity(local_path: str, s3_obj: dict, s3_client, bucket: str, key: str) -> bool:
    """Verify file integrity using S3 ETags."""
    try:
        # Get basic S3 object info
        s3_etag = s3_obj.get("ETag", "").strip('"')
        s3_size = s3_obj.get("Size", 0)
        local_size = os.path.getsize(local_path)
        
        logger.info("File verification - Local: {} bytes, S3: {} bytes (ETag: {})", 
                   local_size, s3_size, s3_etag)
        
        # Check size first - must match
        if local_size != s3_size:
            logger.warning("Size mismatch - Local: {} bytes, S3: {} bytes", local_size, s3_size)
            return False
        
        # Verify using ETag
        if s3_etag:
            # Single-part upload (32 chars, no dash)
            if len(s3_etag) == 32 and "-" not in s3_etag:
                local_md5 = calculate_md5(local_path)
                if local_md5 == s3_etag:
                    logger.debug("Single-part ETag verification successful")
                    return True
                else:
                    logger.warning("Single-part ETag verification failed - Local: {}, S3: {}", local_md5, s3_etag)
                    return False
            
            # Multipart upload (contains dash)
            elif "-" in s3_etag:
                logger.info("Verifying multipart upload (ETag: {})", s3_etag)
                if verify_multipart_etag(local_path, s3_etag):
                    logger.info("Multipart ETag verification successful")
                    return True
                else:
                    logger.warning("Multipart ETag verification failed - file may be incomplete or corrupted")
                    return False
        
        # No ETag available
        logger.warning("No ETag available for verification")
        return False
        
    except Exception as e:
        logger.warning("Error verifying file integrity: {}", e)
        return False


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_multipart_etag(file_path: str, part_size: int = 8 * 1024 * 1024) -> str:
    """Calculate the expected S3 ETag for a multipart upload.
    
    S3 multipart ETags are calculated as:
    MD5(MD5(part1) + MD5(part2) + ... + MD5(partN)) + "-" + part_count
    
    Args:
        file_path: Path to the file
        part_size: Size of each part in bytes (default 8MB, S3 minimum)
    
    Returns:
        Expected ETag string
    """
    import hashlib
    
    file_size = os.path.getsize(file_path)
    part_hashes = []
    
    with open(file_path, "rb") as f:
        part_num = 0
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            part_hashes.append(hashlib.md5(chunk).digest())
            part_num += 1
    
    # Concatenate all part hashes and calculate MD5 of the concatenation
    concatenated = b"".join(part_hashes)
    final_hash = hashlib.md5(concatenated).hexdigest()
    
    return f"{final_hash}-{part_num}"


def find_correct_part_size(file_path: str, s3_etag: str) -> int | None:
    """Find the part size that produces the correct ETag for a multipart upload."""
    if not s3_etag or "-" not in s3_etag:
        return None
    
    try:
        expected_parts = int(s3_etag.split("-")[-1])
    except (ValueError, IndexError):
        return None
    
    file_size = os.path.getsize(file_path)
    
    # Calculate theoretical part size
    theoretical_size = file_size // expected_parts
    
    # Test a range around the theoretical size
    # S3 minimum is 5MB, so start there
    min_size = 5 * 1024 * 1024
    max_size = min(file_size, 64 * 1024 * 1024)  # S3 maximum is 5GB, but let's be reasonable
    
    # Test sizes from min to max in reasonable increments
    test_sizes = []
    for size in range(min_size, max_size + 1, 1024 * 1024):  # 1MB increments
        test_sizes.append(size)
    
    # Also test the theoretical size and nearby values
    for offset in range(-10, 11):
        size = theoretical_size + offset * 1024 * 1024
        if min_size <= size <= max_size:
            test_sizes.append(size)
    
    # Remove duplicates and sort
    test_sizes = sorted(set(test_sizes))
    
    for part_size in test_sizes:
        calculated = calculate_multipart_etag(file_path, part_size)
        if calculated == s3_etag:
            return part_size
    
    return None


def verify_multipart_etag(file_path: str, s3_etag: str) -> bool:
    """Verify a file against a multipart S3 ETag by finding the correct part size."""
    if not s3_etag or "-" not in s3_etag:
        return False
    
    # Find the correct part size that produces the expected ETag
    correct_part_size = find_correct_part_size(file_path, s3_etag)
    
    if correct_part_size:
        logger.debug("Multipart ETag verification successful with part size: {} bytes ({}MB)", 
                    correct_part_size, correct_part_size // 1024 // 1024)
        return True
    else:
        logger.debug("Multipart ETag verification failed - no part size produces the expected ETag")
        return False


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
    skip_checksum: bool = False,
    force_download: bool = False,
    skip_video_processing: bool = False,
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
        
        if force_download:
            # Force re-download regardless of local file existence
            logger.info("Force download enabled, re-downloading {}", key)
            local_path = s3.download_object(src_bucket, key, item_dir)
        elif os.path.exists(local_path) and not skip_checksum:
            # Verify file integrity using checksums
            if verify_file_integrity(local_path, main_file, s3, src_bucket, key):
                logger.info("File {} already exists locally and is verified complete, skipping download", local_path)
            else:
                logger.warning("File {} exists but checksum verification failed, re-downloading", local_path)
                local_path = s3.download_object(src_bucket, key, item_dir)
        elif os.path.exists(local_path) and skip_checksum:
            # Skip checksum verification, use existing file
            logger.info("File {} exists locally, checksum verification disabled, using existing file", local_path)
        else:
            # File doesn't exist locally, download it
            local_path = s3.download_object(src_bucket, key, item_dir)

        mp4_path = os.path.join(work_root, os.path.basename(item_dir) + ".mp4")
        
        # Check if we should skip video processing
        if skip_video_processing:
            # Look for existing MP4 files
            base_name = os.path.splitext(mp4_path)[0]
            mp4_files = []
            
            # Check for single file first
            if os.path.exists(mp4_path):
                mp4_files.append(mp4_path)
            else:
                # Check for multiple part files
                part_num = 1
                while True:
                    part_file = f"{base_name}_part{part_num}.mp4"
                    if os.path.exists(part_file):
                        mp4_files.append(part_file)
                        part_num += 1
                    else:
                        break
            
            if mp4_files:
                logger.info("Skipping video processing - found {} existing MP4 file(s): {}", 
                          len(mp4_files), [os.path.basename(f) for f in mp4_files])
            else:
                logger.warning("Skip video processing enabled but no MP4 files found for {}, skipping", key)
                continue
        else:
            # Normal video processing
            # Check disk space before processing each file
            from .video_extract import check_disk_space
            try:
                check_disk_space(work_root, required_gb=20)
            except RuntimeError as e:
                logger.error("FATAL: {}", e)
                logger.error("Stopping pipeline to prevent further disk space issues")
                import sys
                sys.exit(1)
            
            try:
                logger.info("Extracting and converting main title from {} to {}", local_path, mp4_path)
                extract_main_title_to_mp4(local_path, mp4_path)
                logger.info("Successfully extracted and converted main title")
            except Exception as e:
                # All extraction failures are fatal - stop the pipeline
                logger.error("FATAL: Extraction failed for {}: {}", key, e)
                logger.error("Stopping pipeline due to extraction failure")
                import sys
                sys.exit(1)

            # Find all MP4 files that were created (including multiple parts)
            base_name = os.path.splitext(mp4_path)[0]
            mp4_files = []
            
            # Check for single file first
            if os.path.exists(mp4_path):
                mp4_files.append(mp4_path)
            else:
                # Check for multiple part files
                part_num = 1
                while True:
                    part_file = f"{base_name}_part{part_num}.mp4"
                    if os.path.exists(part_file):
                        mp4_files.append(part_file)
                        part_num += 1
                    else:
                        break
            
            if not mp4_files:
                logger.error("No MP4 files found after extraction for {}", key)
                continue
            
            logger.info("Found {} MP4 file(s) to upload", len(mp4_files))

        # Upload each MP4 file
        for i, mp4_file in enumerate(mp4_files):
            if os.path.exists(mp4_file):
                out_key = os.path.join(output_prefix, os.path.basename(mp4_file)) if output_prefix else os.path.basename(mp4_file)
                
                if dry_run:
                    logger.info("DRY RUN: Would upload {} to s3://{}/{}", mp4_file, dst_bucket, out_key)
                else:
                    s3.upload_file(dst_bucket, mp4_file, out_key)
                    logger.info("Uploaded {} to s3://{}/{}", mp4_file, dst_bucket, out_key)

        # Mark as processed (even in dry run for testing)
        processed_files[key] = {
            "processed_at": datetime.now().isoformat(),
            "output_files": [os.path.basename(f) for f in mp4_files if os.path.exists(f)],
            "size": obj.get("Size", 0),
            "last_modified": obj.get("LastModified", "").isoformat() if obj.get("LastModified") else "",
            "dry_run": dry_run
        }
        save_processed_files(processed_file, processed_files)
        logger.info("Marked {} as processed with {} output files{}", key, len(mp4_files), " (dry run)" if dry_run else "")

        try:
            shutil.rmtree(item_dir, ignore_errors=True)
        except Exception:
            pass


