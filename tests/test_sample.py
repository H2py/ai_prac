"""
Simple test to verify the pipeline setup.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.audio_extractor import AudioExtractor
        from src.utils.audio_utils import load_audio, save_audio
        from src.utils.logger import setup_logger
        from config.settings import Config, AudioConfig
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation."""
    try:
        from config.settings import Config
        
        config = Config()
        config.validate()
        
        print("✅ Configuration created and validated!")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_logger_setup():
    """Test logger setup."""
    try:
        from src.utils.logger import setup_logger
        
        logger = setup_logger("test_logger", level="INFO")
        logger.info("Test message")
        
        print("✅ Logger setup successful!")
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running basic pipeline tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config_creation),
        ("Logger Test", test_logger_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        success = test_func()
        results.append(success)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Summary: {passed}/{total} passed")
    
    if passed == total:
        print("✨ All tests passed! The pipeline is ready to use.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Please check the installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()