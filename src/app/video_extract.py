import os
import subprocess
import shutil
import time
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
        # Use FFmpeg with libdvdcss for direct DVD processing
        try:
            logger.info("Using FFmpeg with libdvdcss for direct DVD processing from ISO")
            
            # Extract VIDEO_TS directory from ISO for FFmpeg processing
            extract_dir = "/data/work/temp_extract"
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            os.makedirs(extract_dir, exist_ok=True)
            
            try:
                # Step 1: Extract VIDEO_TS from ISO using 7z
                logger.info("Extracting VIDEO_TS from ISO using 7z")
                extract_cmd = ["7z", "x", source_path, "-o" + extract_dir, "VIDEO_TS"]
                run(extract_cmd)
                
                # Check if extraction worked
                video_ts_path = os.path.join(extract_dir, "VIDEO_TS")
                if not os.path.exists(video_ts_path):
                    logger.error("VIDEO_TS directory not found after 7z extraction")
                    raise ValueError("VIDEO_TS directory not found in ISO")
                
                # Step 2: Extract DVD title for better naming
                dvd_title = None
                try:
                    # Method 1: Try to get title from VIDEO_TS.IFO file
                    ifo_file = os.path.join(video_ts_path, "VIDEO_TS.IFO")
                    if os.path.exists(ifo_file):
                        logger.info("Attempting to extract DVD title from VIDEO_TS.IFO")
                        try:
                            # Use strings command to extract readable text from IFO file
                            result = subprocess.run(["strings", ifo_file], capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                # Look for common DVD title patterns in the strings output
                                lines = result.stdout.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    # Look for lines that might be DVD titles (reasonable length, no special chars)
                                    if 3 <= len(line) <= 50 and line.isprintable() and not line.isdigit():
                                        # Skip common IFO file strings that aren't titles
                                        if not any(skip in line.lower() for skip in ['video_ts', 'ifo', 'vob', 'menu', 'chapter', 'title', 'angle']):
                                            dvd_title = line
                                            logger.info("Found DVD title from IFO: {}", dvd_title)
                                            break
                        except Exception as e:
                            logger.debug("Failed to extract title from IFO: {}", e)
                    
                    # Method 2: Fallback to ISO filename if IFO method fails
                    if not dvd_title:
                        dvd_title = os.path.splitext(os.path.basename(source_path))[0]
                        logger.info("Using ISO filename as DVD title: {}", dvd_title)
                        
                except Exception as e:
                    logger.warning("Failed to extract DVD title: {}, using ISO filename", e)
                    dvd_title = os.path.splitext(os.path.basename(source_path))[0]
                
                # Step 3: Use FFmpeg with libdvdcss for direct DVD processing
                logger.info("Using FFmpeg with libdvdcss for direct DVD processing")
                
                # Get dynamic encoder settings
                encoder = get_video_encoder(force_encoder)
                preset = get_encoder_preset(force_encoder)
                
                # Build FFmpeg command for DVD processing
                cmd = [
                    "ffmpeg", "-y",
                    "-hide_banner", "-loglevel", "error",
                    "-dvdnav", "1",  # Enable DVD navigation
                    "-i", f"dvd://{video_ts_path}",  # Use dvd:// protocol with VIDEO_TS path
                    "-map", "0:v:0", "-map", "0:a:0?",  # Map first video and audio streams
                    "-c:v", encoder, "-preset", preset,
                    "-r", "29.97",  # Force frame rate for DVD content
                ]
                
                # Add encoder-specific quality settings
                if encoder == "h264_nvenc":
                    # NVENC uses bitrate, not CRF - use more conservative settings for RTX 4090
                    cmd.extend(["-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M", "-rc", "vbr"])
                else:
                    # CPU encoders use CRF
                    cmd.extend(["-crf", "18"])
                
                cmd.extend([
                    "-vf", "yadif=1:1:0",  # Better deinterlacing for DVD content
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    "-max_muxing_queue_size", "1024",
                    output_mp4,
                ])
                
                # Run FFmpeg command
                logger.info("Running FFmpeg DVD processing command")
                processing_start_time = time.time()
                result = run(cmd, capture_errors=True)
                
                if result.returncode != 0:
                    # If NVENC fails, try with CPU encoder as fallback
                    if encoder == "h264_nvenc" and ("unsupported device" in result.stderr or "No capable devices found" in result.stderr or "Error while opening encoder" in result.stderr or "CUDA_ERROR_NO_DEVICE" in result.stderr):
                        logger.warning("NVENC failed, falling back to CPU encoder")
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
                            logger.error("Both NVENC and CPU encoding failed: {}", cpu_result.stderr)
                            raise subprocess.CalledProcessError(cpu_result.returncode, cpu_cmd, cpu_result.stdout, cpu_result.stderr)
                        else:
                            logger.info("Successfully processed with CPU fallback")
                    else:
                        logger.error("FFmpeg DVD processing failed: {}", result.stderr)
                        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                
                processing_elapsed = time.time() - processing_start_time
                logger.info("DVD processing completed in {:.1f}s using FFmpeg with libdvdcss", processing_elapsed)
                
            finally:
                # Clean up extraction directory
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    
        except Exception as e:
            logger.error("FFmpeg DVD processing failed: {}", e)
            raise
    else:
        # Only ISO files are supported
        logger.error("Only ISO files are supported. Found: {}", source_path)
        raise ValueError(f"Unsupported file type: {source_path}. Only ISO files are supported.")


