"""
Speech Emotion Analysis Configuration
설정 파일 - 모든 하이퍼파라미터와 경로들을 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """메인 설정 클래스"""
    
    # 프로젝트 경로
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # 데이터 경로
    RAW_DATA_PATH = DATA_DIR / "raw"
    PROCESSED_DATA_PATH = DATA_DIR / "processed"
    TRAINING_DATA_PATH = DATA_DIR / "training"
    
    # 결과 경로
    TRANSCRIPTIONS_PATH = RESULTS_DIR / "transcriptions"
    EMOTIONS_PATH = RESULTS_DIR / "emotions"
    VISUALIZATIONS_PATH = RESULTS_DIR / "visualizations"
    
    # 디렉토리 생성
    for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, 
                RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_PATH,
                TRANSCRIPTIONS_PATH, EMOTIONS_PATH, VISUALIZATIONS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

class ModelConfig:
    """모델 관련 설정"""
    
    # WhisperX 설정
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v2")
    WHISPER_COMPUTE_TYPE = "float16"  # GPU에서 메모리 절약
    WHISPER_BATCH_SIZE = 16
    
    # OpenSMILE 설정
    OPENSMILE_FEATURE_SET = os.getenv("OPENSMILE_FEATURE_SET", "eGeMAPSv02")
    OPENSMILE_FEATURE_LEVEL = "Functionals"
    
    # Speaker Diarization 설정
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MIN_SPEAKERS = 1
    MAX_SPEAKERS = 10
    
    # 감정 분류 설정
    EMOTION_LABELS = [
        "angry", "disgust", "fear", "happy", 
        "neutral", "sad", "surprise"
    ]
    
    # 오디오 처리 설정
    SAMPLE_RATE = 16000
    MIN_SEGMENT_DURATION = 1.0  # 최소 세그먼트 길이 (초)
    MAX_SEGMENT_DURATION = 30.0  # 최대 세그먼트 길이 (초)

class ProcessingConfig:
    """데이터 처리 관련 설정"""
    
    # 오디오 전처리
    NORMALIZE_AUDIO = True
    REMOVE_SILENCE = True
    SILENCE_THRESHOLD = -40  # dB
    
    # 세그멘테이션
    SEGMENT_OVERLAP = 0.5  # 세그먼트 간 겹침 (초)
    VAD_AGGRESSIVENESS = 2  # Voice Activity Detection 민감도 (0-3)
    
    # 병렬 처리
    NUM_WORKERS = os.cpu_count() // 2
    BATCH_PROCESSING = True

class LoggingConfig:
    """로깅 관련 설정"""
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = Config.LOGS_DIR / "analysis.log"
    ERROR_LOG_FILE = Config.LOGS_DIR / "error.log"
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # 로그 파일 최대 크기 (MB)
    MAX_LOG_SIZE = 10
    BACKUP_COUNT = 5

class AnalysisConfig:
    """분석 관련 설정"""
    
    # 신뢰도 임계값
    MIN_CONFIDENCE_THRESHOLD = 0.3
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    
    # 결과 저장 형식
    SAVE_DETAILED_FEATURES = True
    SAVE_AUDIO_SEGMENTS = False  # 메모리/저장공간 절약을 위해 False
    EXPORT_FORMATS = ["json", "csv"]
    
    # 시각화 설정
    PLOT_EMOTION_TIMELINE = True
    PLOT_SPEAKER_DISTRIBUTION = True
    PLOT_CONFIDENCE_SCORES = True
    
    # 통계 분석
    CALCULATE_SPEAKER_STATS = True
    GENERATE_SUMMARY_REPORT = True

class DevConfig:
    """개발 및 디버깅 설정"""
    
    DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
    VERBOSE_LOGGING = DEBUG_MODE
    
    # 테스트 설정
    TEST_DATA_PATH = Config.PROJECT_ROOT / "tests" / "test_data"
    ENABLE_PERFORMANCE_MONITORING = True
    
    # 프로파일링
    PROFILE_MEMORY_USAGE = DEBUG_MODE
    PROFILE_EXECUTION_TIME = DEBUG_MODE

# 환경별 설정 선택
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # 프로덕션 환경 설정 오버라이드
    ProcessingConfig.NUM_WORKERS = min(os.cpu_count(), 8)
    LoggingConfig.LOG_LEVEL = "WARNING"
    AnalysisConfig.SAVE_DETAILED_FEATURES = False
    DevConfig.DEBUG_MODE = False

elif ENVIRONMENT == "testing":
    # 테스트 환경 설정
    ModelConfig.WHISPER_MODEL_SIZE = "base"  # 빠른 테스트를 위해
    ProcessingConfig.NUM_WORKERS = 2
    LoggingConfig.LOG_LEVEL = "DEBUG"

# 설정 유효성 검사
def validate_config():
    """설정 값들의 유효성을 검사합니다."""
    
    errors = []
    
    # 필수 토큰 확인
    if not ModelConfig.HUGGINGFACE_TOKEN:
        errors.append("HUGGINGFACE_TOKEN이 설정되지 않았습니다. speaker diarization이 작동하지 않을 수 있습니다.")
    
    # 경로 존재 확인
    required_paths = [Config.DATA_DIR, Config.MODELS_DIR]
    for path in required_paths:
        if not path.exists():
            errors.append(f"필수 디렉토리가 존재하지 않습니다: {path}")
    
    # 모델 크기 확인
    valid_whisper_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if ModelConfig.WHISPER_MODEL_SIZE not in valid_whisper_sizes:
        errors.append(f"유효하지 않은 Whisper 모델 크기: {ModelConfig.WHISPER_MODEL_SIZE}")
    
    # 감정 라벨 확인
    if len(ModelConfig.EMOTION_LABELS) < 2:
        errors.append("최소 2개 이상의 감정 라벨이 필요합니다.")
    
    # 워커 수 확인
    if ProcessingConfig.NUM_WORKERS < 1:
        errors.append("NUM_WORKERS는 1 이상이어야 합니다.")
    
    if errors:
        print("설정 오류 발견:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ 모든 설정이 유효합니다.")
    return True

# 설정 정보 출력
def print_config_summary():
    """현재 설정의 요약을 출력합니다."""
    
    print("🔧 현재 설정 요약")
    print("=" * 50)
    print(f"환경: {ENVIRONMENT}")
    print(f"Whisper 모델: {ModelConfig.WHISPER_MODEL_SIZE}")
    print(f"OpenSMILE 특성셋: {ModelConfig.OPENSMILE_FEATURE_SET}")
    print(f"샘플링 레이트: {ModelConfig.SAMPLE_RATE}Hz")
    print(f"워커 수: {ProcessingConfig.NUM_WORKERS}")
    print(f"로그 레벨: {LoggingConfig.LOG_LEVEL}")
    print(f"디버그 모드: {DevConfig.DEBUG_MODE}")
    print(f"HF 토큰 설정: {'✅' if ModelConfig.HUGGINGFACE_TOKEN else '❌'}")
    print("=" * 50)

if __name__ == "__main__":
    print_config_summary()
    validate_config()