#!/usr/bin/env python3
"""Create a test audio file for testing the pipeline."""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio(output_path="test_audio.wav", duration=10):
    """Create a simple test audio file."""
    sample_rate = 16000
    
    # Create time array
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Create a simple signal with varying frequencies (simulating speech)
    signal = np.zeros_like(t)
    
    # Segment 1: 0-3 seconds (lower frequency - speaker 1)
    mask1 = (t >= 0) & (t < 3)
    signal[mask1] = 0.3 * np.sin(2 * np.pi * 200 * t[mask1])
    signal[mask1] += 0.15 * np.sin(2 * np.pi * 400 * t[mask1])
    
    # Segment 2: 3-6 seconds (higher frequency - speaker 2)
    mask2 = (t >= 3) & (t < 6)
    signal[mask2] = 0.3 * np.sin(2 * np.pi * 300 * t[mask2])
    signal[mask2] += 0.15 * np.sin(2 * np.pi * 600 * t[mask2])
    
    # Segment 3: 6-8 seconds (mixed - both speakers)
    mask3 = (t >= 6) & (t < 8)
    signal[mask3] = 0.2 * np.sin(2 * np.pi * 200 * t[mask3])
    signal[mask3] += 0.2 * np.sin(2 * np.pi * 300 * t[mask3])
    
    # Segment 4: 8-10 seconds (lower frequency again)
    mask4 = (t >= 8) & (t < 10)
    signal[mask4] = 0.35 * np.sin(2 * np.pi * 250 * t[mask4])
    
    # Add some noise for realism
    noise = 0.02 * np.random.randn(len(signal))
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save
    sf.write(output_path, signal, sample_rate)
    print(f"✅ Test audio created: {output_path}")
    print(f"   Duration: {duration} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    return Path(output_path)

if __name__ == "__main__":
    create_test_audio()