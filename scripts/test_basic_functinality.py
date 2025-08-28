# scripts/test_basic_functionality.py
import tempfile
import numpy as np
import librosa
from src.analyzer import SpeakerEmotionAnalyzer

def create_test_audio():
    """테스트용 더미 오디오 생성"""
    duration = 5  # 5초
    sample_rate = 16000
    t = np.linspace(0, duration, duration * sample_rate)
    # 간단한 사인파 생성
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz
    
    # 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    librosa.output.write_wav(temp_file.name, audio, sample_rate)
    return temp_file.name

def test_analyzer():
    """기본 분석기 테스트"""
    print("🧪 기본 기능 테스트 시작...")
    
    try:
        # 테스트 오디오 생성
        test_audio = create_test_audio()
        print("✅ 테스트 오디오 생성 완료")
        
        # 분석기 초기화
        analyzer = SpeakerEmotionAnalyzer(
            whisper_model="base",  # 빠른 테스트를 위해 작은 모델
            hf_token=None  # 토큰 없이 테스트
        )
        print("✅ 분석기 초기화 완료")
        
        # 기본 분석 수행
        results = analyzer.analyze_audio(test_audio)
        print(f"✅ 분석 완료: {len(results)}개 세그먼트 발견")
        
        print("🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_analyzer()