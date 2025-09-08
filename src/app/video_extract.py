import os
import subprocess
import shutil
from loguru import logger


def run(cmd: list[str]) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    # Redirect stderr to /dev/null to suppress FFmpeg decoder errors
    with open('/dev/null', 'w') as devnull:
        subprocess.run(cmd, check=True, stderr=devnull)


def check_disk_space(path: str, required_gb: int = 10) -> None:
    """Check if there's enough disk space available."""
    statvfs = os.statvfs(path)
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    
    if free_gb < required_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb}GB required")
    
    logger.debug("Disk space check passed: {:.1f}GB available", free_gb)


def extract_main_title_to_mp4(source_path: str, output_mp4: str) -> None:
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    
    lower = source_path.lower()
    if lower.endswith(".iso"):
        # Try multiple approaches for ISO extraction
        # First try: Use vobcopy to extract main title with CSS decryption
        try:
            temp_dir = os.path.join(os.path.dirname(output_mp4), "temp_vobcopy")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use vobcopy to extract only the main title (title 1) with CSS decryption
            # -i: input ISO, -o: output dir, -1: extract title 1 only, -m: main title
            # -f: force extraction even if vobcopy thinks there's not enough space (it's often wrong)
            vobcopy_cmd = ["vobcopy", "-i", source_path, "-o", temp_dir, "-1", "-m", "-f"]
            logger.info("Running vobcopy command: {}", " ".join(vobcopy_cmd))
            run(vobcopy_cmd)
            logger.info("Vobcopy completed successfully")
            
            # Find all VOB files and process each as a separate video
            vob_files = [f for f in os.listdir(temp_dir) if f.endswith('.vob')]
            logger.info("Found {} VOB files: {}", len(vob_files), vob_files)
            if vob_files:
                # Sort VOB files by name to ensure correct order (e.g., E3_200420-1.vob, E3_200420-2.vob)
                vob_files.sort()
                
                # Process each VOB file as a separate video
                for i, vob_file in enumerate(vob_files):
                    vob_path = os.path.join(temp_dir, vob_file)
                    
                    # Create separate output file for each VOB
                    if len(vob_files) == 1:
                        # Single VOB - use original output name
                        vob_output = output_mp4
                    else:
                        # Multiple VOBs - add suffix (e.g., _part1, _part2)
                        base_name = os.path.splitext(output_mp4)[0]
                        vob_output = f"{base_name}_part{i+1}.mp4"
                    
                    logger.info("Processing VOB file {} as {}", vob_file, os.path.basename(vob_output))
                    
                           cmd = [
                               "ffmpeg", "-y",
                               "-hide_banner", "-loglevel", "quiet",  # Suppress all output
                               "-probesize", "200M", "-analyzeduration", "200M",
                               "-fflags", "+genpts+igndts+ignidx",  # Generate timestamps, ignore DTS and index
                               "-err_detect", "ignore_err",  # Ignore errors and continue
                               "-max_error_rate", "1.0",  # Allow 100% errors (very aggressive)
                               "-threads", "1",  # Single thread to reduce noise
                               "-flags", "+low_delay",  # Low delay mode
                               "-i", vob_path,
                               "-map", "0:v:0", "-map", "0:a:0?",
                               "-c:v", "libx264", "-preset", "slow", "-crf", "18",  # High quality H.264 encoding
                               "-vf", "bwdif=mode=1:parity=auto",  # Auto deinterlacing
                               "-c:a", "aac", "-b:a", "192k",  # AAC audio encoding
                               "-movflags", "+faststart",  # Web optimization
                               "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
                               "-max_muxing_queue_size", "1024",  # Increase muxing queue
                               vob_output,
                           ]
                    run(cmd)
                    logger.info("Successfully processed VOB file {} to {}", vob_file, os.path.basename(vob_output))
                
                # If we created multiple files, we need to return the first one for the pipeline
                # The pipeline will need to be updated to handle multiple outputs
                if len(vob_files) > 1:
                    # Return the first VOB's output for now
                    base_name = os.path.splitext(output_mp4)[0]
                    output_mp4 = f"{base_name}_part1.mp4"
                
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info("Extracted using vobcopy with CSS decryption")
                return
        except Exception as e:
            logger.warning("Vobcopy extraction failed: {}, trying alternative method", e)
        
        # Second try: Use direct FFmpeg conversion (fallback)
        try:
            # Disk space is checked at the pipeline level
            
                   cmd = [
                       "ffmpeg", "-y",
                       "-hide_banner", "-loglevel", "quiet",  # Suppress all output
                       "-probesize", "200M", "-analyzeduration", "200M",
                       "-fflags", "+genpts+igndts+ignidx",  # Generate timestamps, ignore DTS and index
                       "-err_detect", "ignore_err",  # Ignore errors and continue
                       "-max_error_rate", "1.0",  # Allow 100% errors (very aggressive)
                       "-threads", "1",  # Single thread to reduce noise
                       "-flags", "+low_delay",  # Low delay mode
                       "-i", source_path,
                       "-map", "0:v:0", "-map", "0:a:0?",
                       "-c:v", "libx264", "-preset", "slow", "-crf", "18",  # High quality H.264 encoding
                       "-vf", "bwdif=mode=1:parity=auto",  # Auto deinterlacing
                       "-c:a", "aac", "-b:a", "192k",  # AAC audio encoding
                       "-movflags", "+faststart",  # Web optimization
                       "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
                       "-max_muxing_queue_size", "1024",  # Increase muxing queue
                       output_mp4,
                   ]
            run(cmd)
            logger.info("Converted ISO to MP4 with re-encoding")
            return
        except Exception as e:
            logger.error("All ISO extraction methods failed: {}", e)
            raise
    elif "video_ts" in lower:
        # Read from VIDEO_TS folder; pick title 1
        input_url = f"file:dvd://1:{os.path.dirname(source_path) if source_path.lower().endswith('video_ts.ifo') else source_path}"
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "quiet",  # Suppress all output except fatal errors
            "-probesize", "100M", "-analyzeduration", "100M",
            "-fflags", "+genpts+igndts",  # Generate timestamps, ignore DTS
            "-err_detect", "ignore_err",  # Ignore errors and continue
            "-i", input_url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "copy",
            "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
            output_mkv,
        ]
        run(cmd)
    else:
        # Already a file with video
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "quiet",  # Suppress all output except fatal errors
            "-fflags", "+genpts+igndts",  # Generate timestamps, ignore DTS
            "-err_detect", "ignore_err",  # Ignore errors and continue
            "-i", source_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "copy",
            "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
            output_mkv,
        ]
        run(cmd)


