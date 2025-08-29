#!/usr/bin/env python3
"""
Test script for ASS subtitle generation pipeline.
"""

import os
import sys
from pathlib import Path

def test_ass_pipeline():
    """Test the complete pipeline with ASS output."""
    
    print("🎬 Testing ASS Subtitle Pipeline")
    print("=" * 50)
    
    # Check if test audio exists
    test_file = "test_audio.wav"
    if not Path(test_file).exists():
        print("Creating test audio file...")
        os.system("python create_test_audio.py")
    
    print("\n1. Testing without STT (placeholder text):")
    print("-" * 40)
    cmd1 = f"python main.py {test_file} --output ./test_ass --format json --export-ass --no-gpu"
    print(f"Command: {cmd1}")
    os.system(cmd1)
    
    print("\n2. Testing with STT (actual transcription):")
    print("-" * 40)
    cmd2 = f"python main.py {test_file} --output ./test_ass_stt --format json --enable-stt --stt-language ko --export-ass --no-gpu"
    print(f"Command: {cmd2}")
    os.system(cmd2)
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    print("\nGenerated files:")
    
    # Check generated files
    dirs_to_check = ["./test_ass", "./test_ass_stt"]
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            print(f"\n{dir_path}:")
            for file in Path(dir_path).glob("*"):
                print(f"  • {file.name}")
    
    print("\n💡 To use the ASS files:")
    print("  1. Open in video editor (Premiere, DaVinci, etc.)")
    print("  2. Or use with video player (VLC, MPC-HC, etc.)")
    print("  3. Or convert to SRT: ffmpeg -i subtitles.ass subtitles.srt")

if __name__ == "__main__":
    test_ass_pipeline()