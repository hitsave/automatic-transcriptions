import os
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


def run(cmd: list[str], capture_errors: bool = False) -> subprocess.CompletedProcess:
    logger.debug("Running: {}", " ".join(cmd))
    if capture_errors:
        # Capture stderr to see what's failing
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Command failed with exit code {}: {}", result.returncode, " ".join(cmd))
            logger.error("STDOUT: {}", result.stdout)
            logger.error("STDERR: {}", result.stderr)
            # Don't raise exception here - let the calling code handle it
        return result
    else:
        # Redirect stderr to /dev/null to suppress FFmpeg decoder errors
        with open('/dev/null', 'w') as devnull:
            result = subprocess.run(cmd, check=True, stderr=devnull)
        return result


def process_vob_file(vob_path: str, vob_output: str, vob_file: str, encoder: str, preset: str) -> tuple[str, float, float]:
    """Process a single VOB file and return timing information."""
    vob_start_time = time.time()
    logger.info("Processing VOB file {} as {}", vob_file, os.path.basename(vob_output))
    
    ffmpeg_start_time = time.time()
    # First, check if the VOB file has video streams
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", vob_path
    ]
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        has_video = probe_result.returncode == 0 and "video" in probe_result.stdout
        if not has_video:
            logger.warning("VOB file {} has no video streams, skipping", vob_file)
            return vob_file, 0.0, 0.0
    except Exception as e:
        logger.warning("Could not probe VOB file {}: {}", vob_file, e)
        return vob_file, 0.0, 0.0

    # Build command with encoder-specific parameters
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",  # Show errors but not warnings
        "-probesize", "200M", "-analyzeduration", "200M",
        "-fflags", "+genpts+igndts+ignidx",
        "-err_detect", "ignore_err",
        "-max_error_rate", "1.0",
        "-threads", "0",  # Use all available threads for this VOB
        "-flags", "+low_delay",
        "-i", vob_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", encoder, "-preset", preset,
        "-r", "29.97",  # Force frame rate for DVD content
    ]
    
    # Add encoder-specific quality settings
    if encoder == "h264_nvenc":
        # NVENC uses bitrate, not CRF - use more conservative settings for RTX 4090
        cmd.extend(["-b:v", "4M", "-maxrate", "6M", "-bufsize", "8M", "-rc", "vbr"])
    else:
        # CPU encoders use CRF
        cmd.extend(["-crf", "18"])
    
    cmd.extend([
        "-vf", "yadif=1:1:0",  # Better deinterlacing for DVD content
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-max_muxing_queue_size", "1024",
        vob_output,
    ])
    result = run(cmd, capture_errors=True)
    
    # Check if the command failed
    if result.returncode != 0:
        # If NVENC fails, try with CPU encoder as fallback
        if encoder == "h264_nvenc" and ("unsupported device" in result.stderr or "No capable devices found" in result.stderr or "Error while opening encoder" in result.stderr or "CUDA_ERROR_NO_DEVICE" in result.stderr):
            logger.warning("NVENC failed for {}, falling back to CPU encoder", vob_file)
            cpu_cmd = cmd.copy()
            cpu_cmd[cpu_cmd.index("h264_nvenc")] = "libx264"
            cpu_cmd[cpu_cmd.index("p7")] = "slow"
            # Replace NVENC bitrate parameters with CRF for CPU encoder
            if "-b:v" in cpu_cmd:
                bv_index = cpu_cmd.index("-b:v")
                # Remove bitrate, maxrate, bufsize, and rc parameters
                cpu_cmd = cpu_cmd[:bv_index] + ["-crf", "18"] + cpu_cmd[bv_index+8:]
            
            cpu_result = run(cpu_cmd, capture_errors=True)
            if cpu_result.returncode != 0:
                logger.error("Both NVENC and CPU encoding failed for {}: {}", vob_file, cpu_result.stderr)
                raise subprocess.CalledProcessError(cpu_result.returncode, cpu_cmd, cpu_result.stdout, cpu_result.stderr)
            else:
                logger.info("Successfully processed {} with CPU fallback", vob_file)
        else:
            logger.error("Failed to process VOB file {}: {}", vob_file, result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    ffmpeg_elapsed = time.time() - ffmpeg_start_time
    vob_elapsed = time.time() - vob_start_time
    logger.info("Successfully processed VOB file {} to {} (FFmpeg: {:.1f}s, Total: {:.1f}s)", 
              vob_file, os.path.basename(vob_output), ffmpeg_elapsed, vob_elapsed)
    
    return vob_file, ffmpeg_elapsed, vob_elapsed




def is_gpu_available() -> bool:
    """Check if NVIDIA GPU is available for encoding."""
    try:
        # Simple approach: Just check if nvidia-smi works
        # In Docker Desktop WSL2, device files won't be available but nvidia-smi works
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            nvidia_smi_works = result.returncode == 0 and 'NVIDIA' in result.stdout
            logger.debug("nvidia-smi works: {}", nvidia_smi_works)
            
            if not nvidia_smi_works:
                return False
                
            # Check driver version for NVENC support (requires 570.0+)
            import re
            version_match = re.search(r'Driver Version: (\d+)\.(\d+)', result.stdout)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                driver_version = major * 100 + minor
                logger.debug("NVIDIA driver version: {}.{} ({})", major, minor, driver_version)
                if driver_version < 570:
                    logger.warning("NVIDIA driver version {}.{} is too old for NVENC (requires 570.0+)", major, minor)
                    return False
                    
            logger.info("GPU detected via nvidia-smi - allowing GPU usage")
            return True
            
        except Exception as e:
            logger.debug("nvidia-smi not available: {}", e)
            return False
            
    except Exception as e:
        logger.warning("Error in GPU detection: {}", e)
        return False


# Cache GPU detection result to avoid repeated checks
_gpu_available_cache = None

def get_video_encoder(force_encoder: str | None = None) -> str:
    """Get the best available video encoder (GPU or CPU)."""
    # If force_encoder is specified, use it
    if force_encoder == "gpu":
        logger.info("Forcing GPU encoder (h264_nvenc)")
        return "h264_nvenc"
    elif force_encoder == "cpu":
        logger.info("Forcing CPU encoder (libx264)")
        return "libx264"
    
    # Otherwise, use automatic detection
    global _gpu_available_cache
    if _gpu_available_cache is None:
        _gpu_available_cache = is_gpu_available()
        logger.info("GPU detection result: {}", _gpu_available_cache)
    
    if _gpu_available_cache:
        logger.info("GPU detected, using NVENC encoder")
        return "h264_nvenc"
    else:
        logger.info("No GPU detected, using CPU encoder")
        return "libx264"


def get_encoder_preset(force_encoder: str | None = None) -> str:
    """Get the appropriate preset for the selected encoder."""
    # If force_encoder is specified, use appropriate preset
    if force_encoder == "gpu":
        return "p7"  # NVENC preset (p7 = high quality, slowest)
    elif force_encoder == "cpu":
        return "slow"  # x264 preset
    
    # Otherwise, use automatic detection
    global _gpu_available_cache
    if _gpu_available_cache is None:
        _gpu_available_cache = is_gpu_available()
    
    if _gpu_available_cache:
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


def extract_main_title_to_mp4(source_path: str, output_mp4: str, force_encoder: str | None = None) -> None:
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    
    lower = source_path.lower()
    if lower.endswith(".iso"):
        # Use vobcopy directly on ISO file for decryption
        try:
            logger.info("Using vobcopy to decrypt VOB files directly from ISO")
            
            # Create vobcopy output directory
            vobcopy_dir = "/data/work/temp_vobcopy"
            if os.path.exists(vobcopy_dir):
                shutil.rmtree(vobcopy_dir, ignore_errors=True)
            os.makedirs(vobcopy_dir, exist_ok=True)
            
            # Use vobcopy to decrypt the VOB files directly from ISO
            # Use -M for main title (longest) instead of -m for mirror (entire DVD)
            # Use -v for verbose progress and -F for faster processing
            vobcopy_cmd = ["vobcopy", "-i", source_path, "-o", vobcopy_dir, "-M", "-f", "-v", "-F", "4"]
            logger.info("Running vobcopy to decrypt VOB files (this may take several minutes)")
            # Don't suppress stderr for vobcopy so we can see progress
            result = subprocess.run(vobcopy_cmd, check=True)
            
            # Find decrypted VOB files
            vob_files = [f for f in os.listdir(vobcopy_dir) if f.upper().endswith('.VOB')]
            video_ts_path = vobcopy_dir  # Use the decrypted VOB files
                
                if vob_files:
                    logger.info("Found {} VOB files after vobcopy decryption: {}", len(vob_files), vob_files)
                    
                    # Get dynamic encoder settings
                    encoder = get_video_encoder(force_encoder)
                    preset = get_encoder_preset(force_encoder)
                    
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
                    shutil.rmtree(vobcopy_dir, ignore_errors=True)
                    logger.info("Extracted using vobcopy method")
                    return
                else:
                    logger.error("No VOB files found after vobcopy decryption - this is not a valid DVD ISO")
                    raise ValueError("No VOB files found after decryption - not a valid DVD structure")
        except Exception as e:
            logger.error("VOB processing failed: {}", e)
            raise
    else:
        # Only ISO files are supported - use vobcopy method
        logger.error("Only ISO files are supported. Found: {}", source_path)
        raise ValueError(f"Unsupported file type: {source_path}. Only ISO files are supported.")


