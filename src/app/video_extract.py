import os
import subprocess
import shutil
import time
import re
from typing import List, Optional
from loguru import logger
import pytesseract
from PIL import Image

# Import the new advanced DVD menu analysis module
# Advanced DVD menu analysis is available as a separate library
# but not used in the main pipeline to keep it simple
ADVANCED_MENU_ANALYSIS_AVAILABLE = False


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


def get_video_encoder() -> str:
    """Get the video encoder (CPU-only)."""
    logger.info("Using CPU encoder (libx264)")
    return "libx264"


def get_encoder_preset() -> str:
    """Get the appropriate preset for the CPU encoder."""
    return "slow"  # x264 preset


def check_disk_space(path: str, required_gb: int = 10) -> None:
    """Check if there's enough disk space available."""
    statvfs = os.statvfs(path)
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    
    if free_gb < required_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb}GB required")
    
    logger.debug("Disk space check passed: {:.1f}GB available", free_gb)


def _is_likely_menu_frame(image_path: str) -> bool:
    """
    Check if an extracted frame looks like a DVD menu.
    
    This uses simple heuristics to determine if the frame contains menu-like elements.
    """
    try:
        # Basic checks
        if not os.path.exists(image_path):
            return False
        
        file_size = os.path.getsize(image_path)
        if file_size < 1024:  # Too small to be a meaningful frame
            return False
        
        # For now, we'll use a simple heuristic: if the file size is reasonable
        # and the extraction succeeded, we'll assume it might be a menu
        # In a more sophisticated implementation, we could use OpenCV to analyze
        # the image for menu-like elements (buttons, text, etc.)
        
        # Menu frames are typically:
        # - Not too small (content frames might be smaller)
        # - Not too large (very large frames might be full content)
        # - Reasonable size for a DVD frame
        
        if 100000 < file_size < 2000000:  # 100KB to 2MB
            return True
        
        return False
        
    except Exception as e:
        logger.debug("Error checking if frame is menu-like: {}", e)
        return False


def extract_menu_frame_for_ocr(iso_path: str, output_path: str) -> bool:
    """
    Extract a frame from the DVD menu for OCR analysis.
    
    This function specifically targets the DVD menu, not the content titles.
    
    Args:
        iso_path: Path to the ISO file
        output_path: Path to save the extracted frame
        
    Returns:
        True if frame was extracted successfully, False otherwise
    """
    try:
        # First, let's identify the menu VOB files
        # DVD menus are typically in VTS_XX_0.VOB files (where XX is the VTS number)
        # The main menu is usually VTS_01_0.VOB
        
        # Try multiple approaches to access the DVD menu using proper DVD menu extraction
        approaches = [
            # Approach 1: Proper DVD menu extraction with menu parameters
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "1", "-menu_vts", "1", "-pgc", "1", "-pg", "1", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 1, PGC 1, PG 1)"
            },
            # Approach 2: Try VTS 0 (root menu)
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "1", "-menu_vts", "0", "-pgc", "1", "-pg", "1", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 0, PGC 1, PG 1)"
            },
            # Approach 3: Try different PGC values
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "1", "-menu_vts", "1", "-pgc", "0", "-pg", "1", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 1, PGC 0, PG 1)"
            },
            # Approach 4: Try different PG values
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "1", "-menu_vts", "1", "-pgc", "1", "-pg", "0", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 1, PGC 1, PG 0)"
            },
            # Approach 5: Try VTS 2 (if it exists)
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "1", "-menu_vts", "2", "-pgc", "1", "-pg", "1", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 2, PGC 1, PG 1)"
            },
            # Approach 6: Fallback to direct DVD access (like VLC)
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-i", iso_path, "-ss", "0.1", "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "Direct DVD access (fallback)"
            },
            # Approach 7: Try with different language unit
            {
                "cmd": ["ffmpeg", "-y", "-f", "dvdvideo", "-menu", "1", "-menu_lu", "0", "-menu_vts", "1", "-pgc", "1", "-pg", "1", "-i", iso_path, "-vframes", "1", "-vf", "scale=720:480", output_path],
                "description": "DVD menu extraction (VTS 1, LU 0)"
            }
        ]
        
        for approach in approaches:
            logger.debug("Trying menu extraction: {}", approach["description"])
            result = subprocess.run(approach["cmd"], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                # Additional check: try to determine if this looks like a menu
                if _is_likely_menu_frame(output_path):
                    logger.debug("Successfully extracted menu frame using: {}", approach["description"])
                    return True
                else:
                    logger.debug("Extracted frame but doesn't look like a menu: {}", approach["description"])
                    # Continue trying other approaches
            else:
                logger.debug("Failed with {}: {}", approach["description"], result.stderr[:100])
        
        logger.warning("Failed to extract menu frame with any approach")
        return False
            
    except Exception as e:
        logger.error("Error extracting menu frame: {}", e)
        return False


def extract_menu_from_vts_files(iso_path: str) -> dict:
    """
    Try to extract menu information from VTS files directly.
    
    Args:
        iso_path: Path to the ISO file
        
    Returns:
        Dictionary containing menu information
    """
    menu_info = {
        'publishers': [],
        'menu_text': '',
        'success': False
    }
    
    try:
        # Try to extract frames from different VTS files
        vts_files = ['VTS_00_0.VOB', 'VTS_01_0.VOB', 'VTS_02_0.VOB']
        
        for vts_file in vts_files:
            logger.debug("Trying to extract menu from {}", vts_file)
            
            # Try to extract a frame from this VTS file
            cmd = [
                "ffmpeg", "-y",
                "-f", "mpeg",
                "-i", iso_path,
                "-map", "0:0",  # First video stream
                "-ss", "0.5",
                "-vframes", "1",
                "-vf", "scale=720:480",
                f"/data/work/menu_{vts_file.replace('.VOB', '')}.png"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                frame_path = f"/data/work/menu_{vts_file.replace('.VOB', '')}.png"
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 1024:
                    # Try OCR on this frame
                    text = extract_text_from_frame(frame_path)
                    if text and len(text.strip()) > 10:  # More than just a few characters
                        logger.info("Found menu text in {}: {}", vts_file, text[:100])
                        menu_info['menu_text'] = text
                        
                        # Look for publisher patterns
                        publisher_patterns = [
                            r'([A-Z][A-Z\s]+(?:PS2|Xbox|GC|PC|ARCADE|GBA|DS))',
                            r'(ACTIVISION|ATARI|CAPCOM|Eidos|ELECTRONIC ARTS|FROM SOFTWARE|KONAMI|Lucas Arts|MAJESCO|Microsoft|MIDWAY|NAMCO|Nintendo|Sammy Studios|SEGA|Sony Computer Entertainment|THQ|UBISOFT|Vivendi Universal Games)',
                            r'(メーカー別最新映像)',  # Japanese menu header
                        ]
                        
                        for pattern in publisher_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                menu_info['publishers'].extend(matches)
                                logger.debug("Found publishers in {}: {}", vts_file, matches)
                        
                        if menu_info['publishers']:
                            menu_info['success'] = True
                            logger.info("Successfully extracted {} publishers from {}", len(menu_info['publishers']), vts_file)
                            return menu_info
        
        logger.warning("No menu information found in any VTS file")
        return menu_info
        
    except Exception as e:
        logger.error("Error extracting menu from VTS files: {}", e)
        return menu_info


def analyze_dvd_menu(iso_path: str) -> dict:
    """
    Analyze the DVD menu to extract publisher information using simple OCR.
    
    This function uses a simple approach for menu analysis:
    - Extract a menu frame from the DVD
    - Perform full-image OCR on the frame
    - Extract publisher information from the text
    
    For advanced DVD menu analysis (IFO parsing, visual analysis, etc.),
    use the separate dvd_menu_library module.
    
    Args:
        iso_path: Path to the ISO file
        
    Returns:
        Dictionary containing publisher information and menu structure
    """
    menu_info = {
        'publishers': [],
        'menu_text': '',
        'success': False,
        'method': 'simple_ocr'
    }
    
    try:
        # Extract menu frame
        menu_frame = "/data/work/menu_analysis.png"
        if not extract_menu_frame_for_ocr(iso_path, menu_frame):
            logger.warning("Failed to extract menu frame")
            menu_info['method'] = 'extraction_failed'
            return menu_info
        
        # Use Tesseract to extract text from the menu frame
        try:
            menu_text = pytesseract.image_to_string(Image.open(menu_frame), lang='eng+jpn')
            menu_text = menu_text.strip()
            
            if menu_text:
                # Extract publishers from the text
                publishers = extract_publishers_from_text(menu_text)
                
                menu_info.update({
                    'publishers': publishers,
                    'menu_text': menu_text,
                    'success': len(publishers) > 0,
                    'method': 'simple_ocr'
                })
                
                logger.info("Simple OCR analysis successful, found {} publishers", len(publishers))
            else:
                logger.warning("No text extracted from menu frame")
                menu_info['method'] = 'no_text_found'
                
        except Exception as e:
            logger.error("Error in OCR analysis: {}", e)
            menu_info['method'] = 'ocr_error'
    
    except Exception as e:
        logger.error("Error in menu analysis: {}", e)
        menu_info['method'] = 'analysis_error'
    
    return menu_info


def extract_publishers_from_text(text: str) -> List[str]:
    """
    Extract publisher names from menu text using pattern matching.
    
    Args:
        text: Menu text to analyze
        
    Returns:
        List of detected publisher names
    """
    publishers = []
    
    # Common publisher patterns
    publisher_patterns = [
        r'([A-Z][A-Z\s]+(?:PS2|Xbox|GC|PC|ARCADE|GBA|DS))',
        r'(ACTIVISION|ATARI|CAPCOM|Eidos|ELECTRONIC ARTS|FROM SOFTWARE|KONAMI|Lucas Arts|MAJESCO|Microsoft|MIDWAY|NAMCO|Nintendo|Sammy Studios|SEGA|Sony Computer Entertainment|THQ|UBISOFT|Vivendi Universal Games)',
        r'(メーカー別最新映像)',  # Japanese menu header
    ]
    
    for pattern in publisher_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            publishers.extend(matches)
            logger.debug("Found publishers: {}", matches)
    
    return publishers


def get_smart_title_name(iso_path: str, title_num: int, dvd_title: str, menu_info: dict = None) -> str:
    """
    Get a smart title name using OCR analysis.
    
    Args:
        iso_path: Path to the ISO file
        title_num: DVD title number
        dvd_title: Base DVD title
        menu_info: Pre-analyzed menu information (currently unused)
        
    Returns:
        Smart title name
    """
    # Use menu information if available for better naming
    if menu_info and menu_info.get('publishers'):
        # Use publisher information for naming
        publisher = menu_info['publishers'][0] if menu_info['publishers'] else 'Unknown'
        return f"{publisher}_{dvd_title}_title_{title_num:02d}"
    
    # Fallback to default naming
    return f"{dvd_title}_title_{title_num:02d}"


def extract_all_titles_to_mp4(source_path: str, output_dir: str, analyze_menu: bool = False) -> List[str]:
    """
    Extract all DVD titles from an ISO file to MP4 files using proper DVD navigation.
    
    Args:
        source_path: Path to the ISO file
        output_dir: Directory to save the MP4 files
        analyze_menu: Whether to analyze DVD menu for publisher information
    
    Returns:
        List of paths to the created MP4 files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lower = source_path.lower()
    if lower.endswith(".iso"):
        # Use lsdvd and FFmpeg with libdvdcss for direct ISO processing
        try:
            logger.info("Using lsdvd and FFmpeg with libdvdcss for direct ISO processing")
            
            # Step 1: Use lsdvd to analyze DVD structure directly from ISO
            logger.info("Analyzing DVD structure with lsdvd")
            lsdvd_cmd = ["lsdvd", "-v", source_path]
            lsdvd_result = subprocess.run(lsdvd_cmd, capture_output=True, text=True, timeout=60)
            
            if lsdvd_result.returncode != 0:
                logger.warning("lsdvd failed: {}, falling back to VOB processing", lsdvd_result.stderr)
                # Fallback: extract VIDEO_TS and use VOB processing
                return _extract_vob_files_fallback_from_iso(source_path, output_dir, os.path.splitext(os.path.basename(source_path))[0])
            
            # Parse lsdvd output to get title information
            dvd_info = _parse_lsdvd_output(lsdvd_result.stdout)
            logger.info("Found {} titles on DVD", len(dvd_info.get('titles', [])))
            
            # Step 2: Extract DVD title for better naming
            dvd_title = dvd_info.get('disc_title', os.path.splitext(os.path.basename(source_path))[0])
            logger.info("DVD title: {}", dvd_title)
            
            # Step 3: Extract each title using FFmpeg with dvdvideo format directly from ISO
            output_files = []
            encoder = get_video_encoder()
            preset = get_encoder_preset()
            
            for title_info in dvd_info.get('titles', []):
                title_num = title_info.get('title_number', 1)
                duration = title_info.get('duration', 0)
                
                if duration < 10:  # Skip very short titles (likely menus)
                    logger.debug("Skipping short title {} ({}s)", title_num, duration)
                    continue
                
                # Analyze DVD menu first to get publisher information (if enabled)
                if title_num == 1 and analyze_menu:  # Only analyze menu once and only if enabled
                    logger.info("Analyzing DVD menu for publisher information...")
                    menu_info = analyze_dvd_menu(source_path)
                    
                else:
                    menu_info = None
                
                # Get smart title name using OCR
                smart_title = get_smart_title_name(source_path, title_num, dvd_title, menu_info)
                output_filename = f"{smart_title}.mp4"
                output_path = os.path.join(output_dir, output_filename)
                
                logger.info("Extracting title {} (duration: {}s)...", title_num, duration)
                
                try:
                    # Use FFmpeg with dvdvideo format for direct ISO processing
                    cmd = [
                        "ffmpeg", "-y",
                        "-hide_banner", "-loglevel", "error",
                        "-f", "dvdvideo",  # Use dvdvideo format for proper DVD navigation
                        "-probesize", "200M", "-analyzeduration", "200M",
                        "-fflags", "+genpts+igndts+ignidx",
                        "-err_detect", "ignore_err",
                        "-max_error_rate", "1.0",
                        "-i", source_path,  # Use ISO file directly
                        "-title", str(title_num),  # Select specific DVD title
                        "-map", "0:v:0", "-map", "0:a:0?",  # Map first video and audio streams
                        "-c:v", encoder, "-preset", preset,
                        "-r", "29.97",  # Force frame rate for DVD content
                    ]
                        
                    # Add encoder-specific quality settings
                    cmd.extend(["-crf", "18"])
                    
                    cmd.extend([
                        "-vf", "yadif=1:1:0",  # Better deinterlacing for DVD content
                        "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        "-avoid_negative_ts", "make_zero",
                        "-max_muxing_queue_size", "1024",
                        output_path,
                    ])
                    
                    # Run FFmpeg command
                    processing_start_time = time.time()
                    result = run(cmd, capture_errors=True)
                    
                    if result.returncode != 0:
                        logger.warning("Failed to extract title {}: {}", title_num, result.stderr)
                        continue
                    
                    # Check if the output file was created and has reasonable size
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:  # At least 1KB
                        processing_elapsed = time.time() - processing_start_time
                        logger.info("Successfully extracted title {} in {:.1f}s", title_num, processing_elapsed)
                        output_files.append(output_path)
                    else:
                        logger.warning("Title {} produced no output or empty file", title_num)
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        
                except Exception as e:
                    logger.error("Failed to process title {}: {}", title_num, e)
                    continue
            
            if not output_files:
                logger.warning("No titles were successfully extracted. Falling back to VOB file processing.")
                return _extract_vob_files_fallback_from_iso(source_path, output_dir, dvd_title)
            
            logger.info("DVD processing completed. Created {} title files", len(output_files))
            return output_files
                    
        except Exception as e:
            logger.error("DVD processing failed: {}", e)
            raise
    else:
        # Only ISO files are supported
        logger.error("Only ISO files are supported. Found: {}", source_path)
        raise ValueError(f"Unsupported file type: {source_path}. Only ISO files are supported.")


def _parse_lsdvd_output(lsdvd_output: str) -> dict:
    """
    Parse lsdvd output to extract DVD information.
    
    Args:
        lsdvd_output: Raw output from lsdvd command
        
    Returns:
        Dictionary containing DVD information including titles
    """
    dvd_info = {
        'disc_title': '',
        'titles': []
    }
    
    lines = lsdvd_output.split('\n')
    current_title = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Disc Title:'):
            dvd_info['disc_title'] = line.split(':', 1)[1].strip()
        elif line.startswith('Title:'):
            if current_title:
                dvd_info['titles'].append(current_title)
            
            # Extract title number and duration from "Title: 01, Length: 00:03:49.967 Chapters: ..."
            try:
                # Split by comma to get parts
                parts = line.split(',')
                if len(parts) >= 2:
                    # Extract title number from "Title: 01"
                    title_part = parts[0].strip()
                    title_number = int(title_part.split(':')[1].strip())
                    
                    # Extract duration from "Length: 00:03:49.967 Chapters: 03"
                    length_part = parts[1].strip()
                    if length_part.startswith('Length:'):
                        duration_str = length_part[7:].strip()  # Remove "Length:"
                        # Remove "Chapters: XX" part if present
                        if ' Chapters:' in duration_str:
                            duration_str = duration_str.split(' Chapters:')[0]
                        
                        # Convert HH:MM:SS.mmm to seconds
                        time_parts = duration_str.split(':')
                        if len(time_parts) == 3:
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            seconds_parts = time_parts[2].split('.')
                            seconds = int(seconds_parts[0])
                            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                        else:
                            total_seconds = 0
                    else:
                        total_seconds = 0
                else:
                    title_number = len(dvd_info['titles']) + 1
                    total_seconds = 0
                    
                current_title = {
                    'title_number': title_number,
                    'duration': total_seconds
                }
            except (ValueError, IndexError) as e:
                # Fallback if parsing fails
                current_title = {
                    'title_number': len(dvd_info['titles']) + 1,
                    'duration': 0
                }
                
        elif current_title and 'Chapters:' in line:
            # Parse chapter count
            try:
                chapters = int(line.split(':', 1)[1].strip())
                current_title['chapters'] = chapters
            except (ValueError, IndexError):
                current_title['chapters'] = 0
    
    # Add the last title if it exists
    if current_title:
        dvd_info['titles'].append(current_title)
    
    return dvd_info


def _extract_vob_files_fallback_from_iso(iso_path: str, output_dir: str, dvd_title: str) -> List[str]:
    """
    Fallback method to extract VOB files from ISO when DVD title extraction fails.
    This method is deprecated since we now use direct ISO processing with dvdvideo format.
    
    Args:
        iso_path: Path to the ISO file
        output_dir: Directory to save the MP4 files
        dvd_title: Name of the DVD for output files
    
    Returns:
        List of paths to the created MP4 files
    """
    logger.warning("VOB fallback method is deprecated. Direct ISO processing should be used instead.")
    logger.info("Attempting to process ISO directly with FFmpeg dvdvideo format")
    
    # Try to process the ISO directly with FFmpeg as a last resort
    try:
        # Use FFmpeg to process the ISO directly without title selection
        output_files = []
        encoder = get_video_encoder()
        preset = get_encoder_preset()
        
        output_filename = f"{dvd_title}_fallback.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "dvdvideo",
            "-probesize", "200M", "-analyzeduration", "200M",
            "-fflags", "+genpts+igndts+ignidx",
            "-err_detect", "ignore_err",
            "-max_error_rate", "1.0",
            "-i", iso_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", encoder, "-preset", preset,
            "-r", "29.97",
        ]
        
        # Add encoder-specific quality settings
        cmd.extend(["-crf", "18"])
        
        cmd.extend([
            "-vf", "yadif=1:1:0",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-max_muxing_queue_size", "1024",
            output_path,
        ])
        
        result = run(cmd, capture_errors=True)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            logger.info("Successfully created fallback file: {}", output_path)
            output_files.append(output_path)
        else:
            logger.error("Fallback processing failed: {}", result.stderr if result.returncode != 0 else "No output file created")
            
    except Exception as e:
        logger.error("Fallback processing failed: {}", e)
    
    return output_files


def extract_main_title_to_mp4(source_path: str, output_mp4: str, analyze_menu: bool = False) -> None:
    """
    Extract the main title from a DVD ISO file to MP4.
    This is a compatibility wrapper that calls extract_all_titles_to_mp4 and takes the first result.
    """
    output_dir = os.path.dirname(output_mp4)
    output_files = extract_all_titles_to_mp4(source_path, output_dir, analyze_menu)
    
    if not output_files:
        raise ValueError("No titles were extracted from the DVD")
    
    # Rename the first output file to the expected name
    first_output = output_files[0]
    if first_output != output_mp4:
        os.rename(first_output, output_mp4)
        logger.info("Renamed {} to {}", first_output, output_mp4)


