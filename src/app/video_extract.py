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
        
        # Method 1: Check for NVIDIA device files (may not exist in WSL2)
        nvidia_device = os.path.exists('/dev/nvidia0')
        logger.debug("NVIDIA device /dev/nvidia0 exists: {}", nvidia_device)
        
        # Check if we're in WSL2 (device files may be virtualized)
        # WSL2 can be detected by checking for WSL_DISTRO_NAME or WSLENV environment variables
        is_wsl2 = bool(os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSLENV'))
        if is_wsl2:
            logger.debug("Running in WSL2 - device files may be virtualized")
        
        # Method 2: Check for CUDA_VISIBLE_DEVICES environment variable
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        logger.debug("CUDA_VISIBLE_DEVICES: {}", cuda_visible_devices)
        
        # Method 3: Check if nvidia-smi works and driver version
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            nvidia_smi_works = result.returncode == 0 and 'NVIDIA' in result.stdout
            logger.debug("nvidia-smi works: {}", nvidia_smi_works)
            
            # Check driver version for NVENC support (requires 570.0+)
            if nvidia_smi_works:
                import re
                version_match = re.search(r'Driver Version: (\d+)\.(\d+)', result.stdout)
                if version_match:
                    major, minor = int(version_match.group(1)), int(version_match.group(2))
                    driver_version = major * 100 + minor
                    logger.debug("NVIDIA driver version: {}.{} ({})", major, minor, driver_version)
                    if driver_version < 570:
                        logger.warning("NVIDIA driver version {}.{} is too old for NVENC (requires 570.0+)", major, minor)
                        return False
        except:
            nvidia_smi_works = False
            logger.debug("nvidia-smi not available")
        
        # GPU is available if nvidia-smi works (primary method for WSL2/Docker)
        # In WSL2, device files are virtualized but nvidia-smi still works
        if is_wsl2:
            # In WSL2, prioritize nvidia-smi over device files
            gpu_available = nvidia_smi_works or (cuda_visible_devices and cuda_visible_devices != '')
            if nvidia_smi_works and not nvidia_device:
                logger.info("WSL2 detected: Using nvidia-smi for GPU detection (device files virtualized)")
        else:
            # On native Linux, use all methods
            gpu_available = nvidia_smi_works or nvidia_device or (cuda_visible_devices and cuda_visible_devices != '')
        
        if not gpu_available:
            return False
            
        # Check if FFmpeg has NVENC support and can actually use it
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
            nvenc_available = 'h264_nvenc' in result.stdout
            logger.debug("FFmpeg NVENC support: {}", nvenc_available)
            
            if not nvenc_available:
                return False
                
            # Test if NVENC actually works by trying to create a simple test
            # In WSL2, device files may not be available but NVENC still works
            try:
                test_cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                    '-c:v', 'h264_nvenc', '-preset', 'p7', '-f', 'null', '-'
                ]
                test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                nvenc_works = test_result.returncode == 0
                logger.debug("NVENC functionality test: {}", nvenc_works)
                if not nvenc_works:
                    logger.debug("NVENC test stderr: {}", test_result.stderr)
                    # If nvidia-smi works but NVENC test fails due to device access issues,
                    # we should still allow GPU usage as the actual encoding might work
                    if nvidia_smi_works and ('unsupported device' in test_result.stderr or 'No capable devices found' in test_result.stderr):
                        logger.info("NVENC test failed due to device access issues, but nvidia-smi works - allowing GPU usage")
                        return True
                return nvenc_works
            except subprocess.TimeoutExpired:
                logger.warning("NVENC functionality test timed out")
                return False
            except Exception as e:
                logger.warning("NVENC functionality test failed: {}", e)
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg encoder check timed out")
            return False
        except Exception as e:
            logger.warning("Error checking FFmpeg encoders: {}", e)
            return False
            
    except Exception as e:
        logger.warning("Error in GPU detection: {}", e)
        return False


# Cache GPU detection result to avoid repeated checks
_gpu_available_cache = None

def get_video_encoder() -> str:
    """Get the best available video encoder (GPU or CPU)."""
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


def get_encoder_preset() -> str:
    """Get the appropriate preset for the selected encoder."""
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
            logger.error("VOB processing failed: {}", e)
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


