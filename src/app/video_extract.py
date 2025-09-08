import os
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


def run(cmd: list[str], capture_errors: bool = False) -> None:
    logger.debug("Running: {}", " ".join(cmd))
    if capture_errors:
        # Capture stderr to see what's failing
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Command failed with exit code {}: {}", result.returncode, " ".join(cmd))
            logger.error("STDOUT: {}", result.stdout)
            logger.error("STDERR: {}", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    else:
        # Redirect stderr to /dev/null to suppress FFmpeg decoder errors
        with open('/dev/null', 'w') as devnull:
            subprocess.run(cmd, check=True, stderr=devnull)


def process_vob_file(vob_path: str, vob_output: str, vob_file: str, encoder: str, preset: str) -> tuple[str, float, float]:
    """Process a single VOB file and return timing information."""
    vob_start_time = time.time()
    logger.info("Processing VOB file {} as {}", vob_file, os.path.basename(vob_output))
    
    ffmpeg_start_time = time.time()
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "quiet",
        "-probesize", "200M", "-analyzeduration", "200M",
        "-fflags", "+genpts+igndts+ignidx",
        "-err_detect", "ignore_err",
        "-max_error_rate", "1.0",
        "-threads", "0",  # Use all available threads for this VOB
        "-flags", "+low_delay",
        "-i", vob_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", encoder, "-preset", preset, "-crf", "18",
        "-vf", "bwdif=mode=1:parity=auto",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-max_muxing_queue_size", "1024",
        vob_output,
    ]
    try:
        run(cmd, capture_errors=True)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to process VOB file {}: {}", vob_file, e)
        raise
    
    ffmpeg_elapsed = time.time() - ffmpeg_start_time
    vob_elapsed = time.time() - vob_start_time
    logger.info("Successfully processed VOB file {} to {} (FFmpeg: {:.1f}s, Total: {:.1f}s)", 
              vob_file, os.path.basename(vob_output), ffmpeg_elapsed, vob_elapsed)
    
    return vob_file, ffmpeg_elapsed, vob_elapsed




def is_gpu_available() -> bool:
    """Check if NVIDIA GPU is available for encoding."""
    try:
        # Check multiple ways to detect GPU availability
        gpu_available = False
        
        # Method 1: Check for NVIDIA device files
        nvidia_device = os.path.exists('/dev/nvidia0')
        logger.debug("NVIDIA device /dev/nvidia0 exists: {}", nvidia_device)
        
        # Method 2: Check for CUDA_VISIBLE_DEVICES environment variable
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        logger.debug("CUDA_VISIBLE_DEVICES: {}", cuda_visible_devices)
        
        # Method 3: Check if nvidia-smi works
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            nvidia_smi_works = result.returncode == 0 and 'NVIDIA' in result.stdout
            logger.debug("nvidia-smi works: {}", nvidia_smi_works)
        except:
            nvidia_smi_works = False
            logger.debug("nvidia-smi not available")
        
        # GPU is available if any method succeeds
        gpu_available = nvidia_device or (cuda_visible_devices and cuda_visible_devices != '') or nvidia_smi_works
        
        if not gpu_available:
            return False
            
        # Check if FFmpeg has NVENC support
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
            nvenc_available = 'h264_nvenc' in result.stdout
            logger.debug("FFmpeg NVENC support: {}", nvenc_available)
            return nvenc_available
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg encoder check timed out")
            return False
        except Exception as e:
            logger.warning("Error checking FFmpeg encoders: {}", e)
            return False
            
    except Exception as e:
        logger.warning("Error in GPU detection: {}", e)
        return False


def get_video_encoder() -> str:
    """Get the best available video encoder (GPU or CPU)."""
    gpu_available = is_gpu_available()
    logger.info("GPU detection result: {}", gpu_available)
    
    if gpu_available:
        logger.info("GPU detected, using NVENC encoder")
        return "h264_nvenc"
    else:
        logger.info("No GPU detected, using CPU encoder")
        return "libx264"


def get_encoder_preset() -> str:
    """Get the appropriate preset for the selected encoder."""
    if is_gpu_available():
        return "p7"  # NVENC preset (p7 = high quality, slowest)
    else:
        return "slow"  # x264 preset


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
        # Use 7z to extract VIDEO_TS from ISO, then process VOB files directly
        try:
            # Extract VIDEO_TS from ISO using 7z
            extract_dir = os.path.join(os.path.dirname(output_mp4), "temp_extract")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            os.makedirs(extract_dir, exist_ok=True)
            
            logger.info("Extracting VIDEO_TS from ISO using 7z")
            extract_cmd = ["7z", "x", source_path, "-o" + extract_dir, "VIDEO_TS"]
            run(extract_cmd)
            
            # Use the VOB files directly from 7z extraction (no need for vobcopy)
            video_ts_path = os.path.join(extract_dir, "VIDEO_TS")
            if os.path.exists(video_ts_path):
                logger.info("Using VOB files directly from 7z extraction")
                
                # Find VOB files in the 7z-extracted VIDEO_TS directory
                vob_files = [f for f in os.listdir(video_ts_path) if f.upper().endswith('.VOB')]
                
                if vob_files:
                    logger.info("Found {} VOB files after 7z extraction: {}", len(vob_files), vob_files)
                    
                    # Get dynamic encoder settings
                    encoder = get_video_encoder()
                    preset = get_encoder_preset()
                    
                    # Prepare VOB processing tasks
                    vob_files.sort()
                    processing_start_time = time.time()
                    
                    # Determine optimal thread count (use CPU cores, but limit to avoid overwhelming)
                    import multiprocessing
                    max_workers = min(len(vob_files), multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
                    logger.info("Processing {} VOB files using {} parallel workers", len(vob_files), max_workers)
                    
                    # Process VOB files in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all VOB processing tasks
                        future_to_vob = {}
                        for i, vob_file in enumerate(vob_files):
                            vob_path = os.path.join(video_ts_path, vob_file)
                            
                            if len(vob_files) == 1:
                                vob_output = output_mp4
                            else:
                                base_name = os.path.splitext(output_mp4)[0]
                                vob_output = f"{base_name}_part{i+1}.mp4"
                            
                            future = executor.submit(process_vob_file, vob_path, vob_output, vob_file, encoder, preset)
                            future_to_vob[future] = vob_file
                        
                        # Wait for all tasks to complete
                        completed_count = 0
                        for future in as_completed(future_to_vob):
                            vob_file = future_to_vob[future]
                            try:
                                processed_file, ffmpeg_elapsed, vob_elapsed = future.result()
                                completed_count += 1
                                logger.info("Completed {}/{} VOB files: {} (FFmpeg: {:.1f}s, Total: {:.1f}s)", 
                                          completed_count, len(vob_files), processed_file, ffmpeg_elapsed, vob_elapsed)
                            except Exception as e:
                                logger.error("Failed to process VOB file {}: {}", vob_file, e)
                                raise
                    
                    processing_elapsed = time.time() - processing_start_time
                    logger.info("All {} VOB files processed in {:.1f}s (parallel processing)", len(vob_files), processing_elapsed)
                    
                    # Clean up
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    logger.info("Extracted using 7z method")
                    return
                else:
                    logger.warning("No VOB files found after 7z extraction")
            else:
                logger.warning("VIDEO_TS directory not found after 7z extraction")
        except Exception as e:
            logger.warning("7z extraction method failed: {}, trying direct FFmpeg conversion", e)
        
        # Third try: Use direct FFmpeg conversion (final fallback)
        try:
            # Disk space is checked at the pipeline level
            
            # Get dynamic encoder settings
            encoder = get_video_encoder()
            preset = get_encoder_preset()
            
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
                "-c:v", encoder, "-preset", preset, "-crf", "18",  # Dynamic encoder selection
                "-vf", "bwdif=mode=1:parity=auto",  # Auto deinterlacing
                "-c:a", "aac", "-b:a", "192k",  # AAC audio encoding
                "-movflags", "+faststart",  # Web optimization
                "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
                "-max_muxing_queue_size", "1024",  # Increase muxing queue
                output_mp4,
            ]
            try:
                run(cmd, capture_errors=True)
            except subprocess.CalledProcessError as e:
                logger.error("Direct FFmpeg conversion failed: {}", e)
                raise
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


