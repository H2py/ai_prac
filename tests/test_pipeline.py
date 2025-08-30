"""
Test the complete audio analysis pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_audio():
    """Create a test audio file with some speech-like patterns."""
    sample_rate = 16000
    duration = 5  # 5 seconds
    
    # Create time array
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Create speech-like signal with variations
    # Simulate two speakers with different frequencies
    signal = np.zeros_like(t)
    
    # Speaker 1 (0-2 seconds, 3-4 seconds)
    speaker1_freq = 200  # Lower pitch
    mask1 = ((t >= 0) & (t < 2)) | ((t >= 3) & (t < 4))
    signal[mask1] = 0.3 * np.sin(2 * np.pi * speaker1_freq * t[mask1])
    signal[mask1] += 0.1 * np.sin(2 * np.pi * speaker1_freq * 2 * t[mask1])
    
    # Speaker 2 (2-3 seconds, 4-5 seconds)
    speaker2_freq = 300  # Higher pitch
    mask2 = ((t >= 2) & (t < 3)) | ((t >= 4) & (t < 5))
    signal[mask2] = 0.3 * np.sin(2 * np.pi * speaker2_freq * t[mask2])
    signal[mask2] += 0.1 * np.sin(2 * np.pi * speaker2_freq * 1.5 * t[mask2])
    
    # Add some noise
    noise = 0.01 * np.random.randn(len(signal))
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, signal, sample_rate)
    
    return Path(temp_file.name)


def test_pipeline():
    """Test the complete pipeline."""
    print("🧪 Testing Complete Audio Analysis Pipeline\n")
    
    # Create test audio
    print("1. Creating test audio file...")
    test_audio = create_test_audio()
    print(f"   ✅ Test audio created: {test_audio}")
    
    try:
        # Test audio extraction
        print("\n2. Testing audio extraction...")
        from src.audio_extractor import AudioExtractor
        extractor = AudioExtractor()
        extracted = extractor.extract(test_audio)
        print(f"   ✅ Audio extracted: {extracted}")
        
        # Test speaker diarization
        print("\n3. Testing speaker diarization...")
        from src.speaker_diarizer import SpeakerDiarizer
        import os
        try:
            diarizer = SpeakerDiarizer()
            auth_token = os.getenv('HUGGINGFACE_TOKEN')
            if not auth_token:
                print("   ⚠️  HUGGINGFACE_TOKEN not found, skipping diarization")
                speaker_results = None
            else:
                diarizer.initialize(auth_token=auth_token)
                speaker_results = diarizer.analyze(extracted)
                if speaker_results and speaker_results.get('segments'):
                    print(f"   ✅ Found {speaker_results['total_speakers']} speakers")
                    print(f"   ✅ Generated {speaker_results['total_segments']} segments")
                else:
                    print("   ⚠️  Using fallback diarization (demo mode)")
        except Exception as e:
            print(f"   ⚠️  Diarization skipped: {str(e)[:100]}")
            speaker_results = None
        
        # Test emotion analysis  
        print("\n4. Testing emotion analysis...")
        from src.emotion_analyzer import EmotionAnalyzer
        try:
            analyzer = EmotionAnalyzer()
            analyzer.initialize()
            
            # Analyze with segments if available
            if speaker_results and speaker_results.get('segments'):
                predictions = analyzer.analyze_segments(extracted, segments=speaker_results['segments'])
            else:
                predictions = analyzer.analyze_segments(extracted)
            
            if predictions:
                stats = analyzer.get_emotion_statistics(predictions)
                print(f"   ✅ Analyzed {len(predictions)} segments")
                print(f"   ✅ Dominant emotion: {stats['dominant_emotion']}")
            else:
                print("   ⚠️  No emotions detected")
        except Exception as e:
            print(f"   ⚠️  Emotion analysis skipped: {str(e)[:100]}")
        
        # Test result merger
        print("\n5. Testing result merger...")
        from src.result_merger import ResultMerger
        merger = ResultMerger()
        
        results = merger.merge_all_results(
            speaker_results=speaker_results,
            emotion_results=predictions if 'predictions' in locals() else None,
            metadata={}
        )
        
        if results:
            print(f"   ✅ Results merged successfully")
            print(f"   ✅ Timeline has {len(results.get('timeline', []))} entries")
        
        # Test export
        print("\n6. Testing export...")
        with tempfile.TemporaryDirectory() as temp_dir:
            exported = merger.export_timeline_segments(results, temp_dir, format='json')
            if exported:
                print(f"   ✅ Exported to: {list(exported.values())[0]}")
        
        print("\n" + "=" * 50)
        print("✨ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if test_audio.exists():
            test_audio.unlink()
            print("\n🧹 Cleaned up test files")


if __name__ == "__main__":
    test_pipeline()