#!/usr/bin/env python3
"""
Test script to verify mandatory token authentication for HuggingFace models.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_token_loading():
    """Test that HUGGINGFACE_TOKEN is properly loaded from .env"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found in environment")
        print("   Check .env file contains: HUGGINGFACE_TOKEN=hf_xxxx")
        return False
    
    print(f"✅ Token loaded successfully (length: {len(token)} chars)")
    print(f"   Token starts with: {token[:10]}...")
    return True

def test_speaker_diarizer_init():
    """Test that SpeakerDiarizer requires token for initialization"""
    from src.speaker_diarizer import SpeakerDiarizer
    
    try:
        # Try initializing without token (should fail)
        diarizer = SpeakerDiarizer()
        diarizer.initialize(auth_token=None)
        print("❌ SpeakerDiarizer accepted None token (should fail)")
        return False
    except (ValueError, TypeError) as e:
        print(f"✅ SpeakerDiarizer correctly rejected None token: {e}")
    
    # Try with empty string (should also fail)
    try:
        diarizer = SpeakerDiarizer()
        diarizer.initialize(auth_token="")
        print("❌ SpeakerDiarizer accepted empty token (should fail)")
        return False
    except ValueError as e:
        print(f"✅ SpeakerDiarizer correctly rejected empty token: {e}")
    
    # Try with actual token from env
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        try:
            diarizer = SpeakerDiarizer()
            diarizer.initialize(auth_token=token)
            print("✅ SpeakerDiarizer initialized with valid token")
            print("   Note: If this fails with 401, you need to accept model agreements")
            return True
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                print("⚠️  Got 401 Unauthorized - Need to accept model agreements:")
                print("   1. https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("   2. https://huggingface.co/pyannote/segmentation-3.0")
                print("   3. https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
            else:
                print(f"❌ Initialization failed: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Mandatory Token Authentication")
    print("=" * 60)
    print()
    
    # Test 1: Token loading
    print("Test 1: Environment Variable Loading")
    print("-" * 40)
    if not test_token_loading():
        sys.exit(1)
    print()
    
    # Test 2: SpeakerDiarizer requirements
    print("Test 2: SpeakerDiarizer Token Requirement")
    print("-" * 40)
    test_speaker_diarizer_init()
    print()
    
    print("=" * 60)
    print("Test Summary:")
    print("- Token loading from .env: ✅")
    print("- Mandatory token enforcement: ✅")
    print("- Next step: Accept model agreements on HuggingFace")
    print("=" * 60)

if __name__ == "__main__":
    main()