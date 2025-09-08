#!/usr/bin/env python3
import os
import subprocess
import sys

def check_nvidia_device():
    """Check if NVIDIA device exists"""
    try:
        result = os.path.exists('/dev/nvidia0')
        print(f"NVIDIA device /dev/nvidia0 exists: {result}")
        if not result:
            print("Available devices in /dev:")
            for f in os.listdir('/dev'):
                if 'nvidia' in f.lower():
                    print(f"  Found: /dev/{f}")
        return result
    except Exception as e:
        print(f"Error checking NVIDIA device: {e}")
        return False

def check_ffmpeg_nvenc():
    """Check if FFmpeg has NVENC support"""
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
        nvenc_available = 'h264_nvenc' in result.stdout
        print(f"FFmpeg NVENC support: {nvenc_available}")
        if nvenc_available:
            print("Available NVENC encoders:")
            for line in result.stdout.split('\n'):
                if 'nvenc' in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("FFmpeg encoders output (first 20 lines):")
            for line in result.stdout.split('\n')[:20]:
                print(f"  {line}")
        return nvenc_available
    except Exception as e:
        print(f"Error checking FFmpeg NVENC: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
        return cuda_available
    except ImportError:
        print("PyTorch not available")
        return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

def main():
    print("=== GPU Detection Debug ===")
    print()
    
    nvidia_device = check_nvidia_device()
    print()
    
    ffmpeg_nvenc = check_ffmpeg_nvenc()
    print()
    
    cuda_available = check_cuda()
    print()
    
    print("=== Summary ===")
    print(f"NVIDIA device: {nvidia_device}")
    print(f"FFmpeg NVENC: {ffmpeg_nvenc}")
    print(f"CUDA available: {cuda_available}")
    print(f"GPU should be available: {nvidia_device and ffmpeg_nvenc}")

if __name__ == "__main__":
    main()
