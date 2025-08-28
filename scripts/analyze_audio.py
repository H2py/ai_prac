#!/usr/bin/env python3
"""
단일 오디오 파일 분석 스크립트
Usage: python scripts/analyze_audio.py --input audio.wav --output results/
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ModelConfig, LoggingConfig

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LoggingConfig.LOG_FILE)
        ]
    )

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="WhisperX + OpenSMILE 기반 화자별 감정 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 사용법:
  python scripts/analyze_audio.py --input data/raw/meeting.wav
  python scripts/analyze_audio.py --input audio.wav --output custom_results/
  python scripts/analyze_audio.py --input audio.wav --model large --no-diarization
  python scripts/analyze_audio.py --input audio.wav --emotion-model models/custom.pkl
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="분석할 오디오 파일 경로"
    )
    
    # 선택 인자
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="결과 저장 디렉토리 (기본값: results/)"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=ModelConfig.WHISPER_MODEL_SIZE,
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
        help=f"WhisperX 모델 크기 (기본값: {ModelConfig.WHISPER_MODEL_SIZE})"
    )
    
    parser.add_argument(
        '--emotion-model',
        type=str,
        help="사전 학습된 감정 분류 모델 경로"
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        help="HuggingFace 토큰 (speaker diarization용)"
    )
    
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help="Speaker diarization 건너뛰기"
    )
    
    parser.add_argument