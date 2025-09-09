"""
Advanced DVD Menu Analysis Library

This library provides sophisticated DVD menu detection and analysis capabilities:
- IFO file parsing for menu button coordinates
- Visual analysis using computer vision
- Targeted OCR on specific regions
- Navigation mapping to DVD titles/chapters

This is a specialized library for advanced DVD menu analysis and is not used
in the main video extraction pipeline.
"""

import os
import tempfile
import subprocess
import shutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image
import pytesseract
from loguru import logger

# Import the DVD menu analysis classes
from .dvd_menu_analysis import (
    DVDMenuRegion,
    IFOParser,
    VisualMenuAnalyzer,
    TargetedOCRExtractor,
    AdvancedDVDMenuAnalyzer
)


class DVDMenuLibrary:
    """
    Advanced DVD Menu Analysis Library
    
    This library provides comprehensive DVD menu analysis capabilities
    including IFO parsing, visual analysis, and navigation mapping.
    """
    
    def __init__(self):
        self.visual_analyzer = VisualMenuAnalyzer()
        self.ocr_extractor = TargetedOCRExtractor()
        self.advanced_analyzer = None
    
    def analyze_dvd_menu(self, iso_path: str, menu_frame_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive DVD menu analysis.
        
        Args:
            iso_path: Path to the DVD ISO file
            menu_frame_path: Path to the extracted menu frame image
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.advanced_analyzer:
            self.advanced_analyzer = AdvancedDVDMenuAnalyzer(iso_path)
        
        return self.advanced_analyzer.analyze_dvd_menu(iso_path, menu_frame_path)
    
    def extract_menu_frame(self, iso_path: str, output_path: str) -> bool:
        """
        Extract a menu frame from a DVD ISO using advanced techniques.
        
        Args:
            iso_path: Path to the DVD ISO file
            output_path: Path where the menu frame should be saved
            
        Returns:
            True if menu frame was extracted successfully
        """
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
        
        for i, approach in enumerate(approaches, 1):
            try:
                logger.debug("Trying approach {}: {}", i, approach["description"])
                
                result = subprocess.run(
                    approach["cmd"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 1024:  # Basic validation
                        logger.info("Successfully extracted menu frame using approach {}: {} ({} bytes)", 
                                   i, approach["description"], file_size)
                        return True
                    else:
                        logger.debug("Approach {} produced too small file: {} bytes", i, file_size)
                else:
                    logger.debug("Approach {} failed: {}", i, result.stderr.strip())
                    
            except subprocess.TimeoutExpired:
                logger.debug("Approach {} timed out", i)
            except Exception as e:
                logger.debug("Approach {} error: {}", i, e)
        
        logger.warning("All menu extraction approaches failed")
        return False
    
    def is_likely_menu_frame(self, image_path: str) -> bool:
        """
        Check if an extracted frame looks like a DVD menu.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if the frame appears to be a menu
        """
        try:
            if not os.path.exists(image_path):
                return False
            
            file_size = os.path.getsize(image_path)
            if file_size < 1024:  # Too small to be a meaningful frame
                return False
            
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


# Convenience functions for external use
def analyze_dvd_menu_advanced(iso_path: str, menu_frame_path: str) -> Dict[str, Any]:
    """
    Convenience function for advanced DVD menu analysis.
    
    Args:
        iso_path: Path to the DVD ISO file
        menu_frame_path: Path to the extracted menu frame image
        
    Returns:
        Dictionary containing analysis results
    """
    library = DVDMenuLibrary()
    return library.analyze_dvd_menu(iso_path, menu_frame_path)


def extract_dvd_menu_frame(iso_path: str, output_path: str) -> bool:
    """
    Convenience function for extracting DVD menu frames.
    
    Args:
        iso_path: Path to the DVD ISO file
        output_path: Path where the menu frame should be saved
        
    Returns:
        True if menu frame was extracted successfully
    """
    library = DVDMenuLibrary()
    return library.extract_menu_frame(iso_path, output_path)
