# 🎵 음성 분석 파이프라인

오디오/비디오 파일 및 YouTube URL에서 화자, 감정, 음향 특징을 추출하는 종합 음성 분석 파이프라인입니다.

## 기능

- **오디오 추출**: 오디오 파일, 비디오 파일, YouTube URL에서 추출
- **화자 분리**: 서로 다른 화자 식별 및 분리
- **감정 인식**: 음성에서 7가지 기본 감정 감지
- **음향 분석**: 피치, 에너지, 스펙트럼 특징 추출
- **음성 인식 (STT)**: Whisper를 사용한 자동 음성-텍스트 변환 🆕
- **자막 생성**: ASS 포맷 자막 파일 자동 생성 🆕
- **다양한 출력 형식**: JSON, CSV, ASS 자막, 사람이 읽을 수 있는 보고서
- **시각화**: 파형, 스펙트로그램, 타임라인 플롯 생성

## 설치

```bash
# 의존성 패키지 설치
pip install -r requirements.txt

# 시각화 지원 (선택사항)
pip install matplotlib seaborn
```

## 시작하기

### 빠른 테스트

```bash
# 1. 테스트 오디오 파일 생성
python create_test_audio.py

# 2. 분석 실행
python main.py test_audio.wav

# 3. output/ 폴더에서 결과 확인
cat output/report_*.txt
```

### 사용자 파일 사용하기

오디오 파일은 **어디에든** 배치 가능합니다 - 현재 디렉토리, 하위 디렉토리, 또는 전체 경로 사용:

```bash
# 현재 디렉토리
python main.py audio.wav

# 하위 디렉토리
python main.py input/audio.wav

# 전체 경로
python main.py /Users/you/Downloads/audio.wav

# 비디오 파일 (자동으로 오디오 추출)
python main.py video.mp4

# YouTube URL
python main.py "https://youtube.com/watch?v=VIDEO_ID"
```

### 권장 폴더 구조

```
ai_prac/
├── input/          # 선택사항: 오디오/비디오 파일을 여기에 배치
│   └── audio.wav
├── output/         # 결과가 자동으로 여기에 저장됨
│   ├── extracted_audio_*.wav
│   ├── timeline_*.json
│   ├── timeline_*.csv
│   └── report_*.txt
└── main.py
```

## 사용 예제

### 기본 분석
```bash
python main.py audio.wav
```

### 옵션 사용
```bash
# CSV로 내보내기
python main.py audio.wav --format csv

# JSON과 CSV 모두
python main.py audio.wav --format both

# 시각화 포함
python main.py audio.wav --verbose

# 사용자 지정 출력 디렉토리
python main.py audio.wav --output ./my_results

# 상세 모드 (진행 상황 표시)
python main.py audio.wav --verbose

# 음성 인식 + 자막 생성 🆕
python main.py audio.wav --enable-stt --export-ass

# 한국어 음성 인식 🆕
python main.py audio.wav --enable-stt --stt-language ko --export-ass
```

### 전체 분석
```bash
python main.py audio.wav \
    --output ./results \
    --format both \
    --enable-stt \
    --export-ass \
    --verbose
```

## 출력 파일

| 파일 | 설명 |
|------|------|
| `report_*.txt` | 사람이 읽을 수 있는 요약 |
| `timeline_*.json` | 상세 분석 데이터 |
| `timeline_*.csv` | 스프레드시트 형식 |
| `subtitles_*.ass` | ASS 자막 파일 (--export-ass 사용 시) 🆕 |
| `extracted_audio_*.wav` | 처리된 오디오 |

### 결과 이해하기

**report_*.txt** 포함 내용:
- 오디오 정보 (길이, 샘플 레이트)
- 화자 분포 (누가 얼마나 말했는지)
- 감정 분포 (감지된 감정들)
- 타임라인 요약

**timeline_*.json** 구조:
```json
{
  "metadata": {
    "input_source": "audio.wav",
    "duration": 120.5
  },
  "speakers": {
    "speaker_1": {"speaking_percentage": 60.0}
  },
  "emotions": {
    "dominant_emotion": "neutral",
    "emotion_distribution": {
      "happy": 30.0,
      "neutral": 50.0,
      "sad": 20.0
    }
  },
  "timeline": [
    {
      "start": 0.0,
      "end": 5.0,
      "speaker": "speaker_1",
      "emotion": "neutral"
    }
  ]
}
```

## 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--output, -o` | 출력 디렉토리 | `./output` |
| `--format, -f` | 출력 형식 (json/csv/both) | `json` |
| `--enable-stt` | 음성 인식 활성화 🆕 | False |
| `--stt-language` | STT 언어 (ko, en, auto) 🆕 | auto |
| `--export-ass` | ASS 자막 파일 생성 🆕 | False |
| `--require-all` | 모든 단계 필수 (실패 시 중단) 🆕 | False |
| `--verbose, -v` | 진행 상황 표시 | False |
| `--gpu/--no-gpu` | GPU 가속 | `--gpu` |
| `--sample-rate` | 샘플 레이트 (Hz) | 16000 |

모든 옵션은 `python main.py --help`를 실행하여 확인하세요.

## 지원 형식

**입력:**
- 오디오: WAV, MP3, FLAC, M4A, OGG, AAC
- 비디오: MP4, AVI, MOV, MKV, WebM
- URL: YouTube 및 기타 플랫폼

**출력:**
- JSON (구조화된 데이터)
- CSV (스프레드시트)
- TXT (보고서)
- PNG (시각화)

## Python API

```python
from src.audio_extractor import AudioExtractor
from src.speaker_diarizer import SpeakerDiarizer
from src.emotion_analyzer import EmotionAnalyzer
from src.result_merger import ResultMerger

# 오디오 추출
extractor = AudioExtractor()
audio_path = extractor.extract("video.mp4")

# 화자 분석
diarizer = SpeakerDiarizer()
speakers = diarizer.analyze(audio_path)

# 감정 분석
analyzer = EmotionAnalyzer()
emotions = analyzer.analyze_segments(audio_path)

# 결과 병합
merger = ResultMerger()
results = merger.merge_all_results(
    speaker_results=speakers,
    emotion_results=emotions
)

# 내보내기
merger.export_to_json(results, "results.json")
```

## 문제 해결

### 누락된 의존성
```bash
pip install -r requirements.txt
```

### 화자/감정이 감지되지 않음
파이프라인은 모델을 사용할 수 없을 때 대체 방법을 포함합니다. 전체 기능을 사용하려면:
1. HuggingFace 토큰 받기: https://huggingface.co/settings/tokens
2. 설정: `export HF_AUTH_TOKEN="your_token"`

### 메모리 문제
큰 파일의 경우:
```bash
python main.py large_file.wav --segment-duration 10
```

### FFmpeg를 찾을 수 없음
FFmpeg 설치:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

## 프로젝트 구조

```
ai_prac/
├── src/
│   ├── audio_extractor.py      # 오디오 추출
│   ├── speaker_diarizer.py     # 화자 식별
│   ├── emotion_analyzer.py     # 감정 감지
│   ├── acoustic_analyzer.py    # 음향 특징
│   ├── result_merger.py        # 결과 결합
│   └── utils/
│       ├── audio_utils.py      # 오디오 처리
│       ├── visualization.py    # 플로팅
│       └── logger.py           # 로깅
├── config/
│   └── settings.py             # 설정
├── main.py                     # CLI 인터페이스
└── requirements.txt            # 의존성 패키지
```

## 요구사항

- Python 3.8+
- 핵심: numpy, librosa, soundfile
- ML: torch, transformers, pyannote.audio
- 미디어: moviepy, yt-dlp
- 선택사항: matplotlib, seaborn (시각화용)

## 라이선스

MIT License

## 기여하기

기여를 환영합니다! Pull Request를 제출해 주세요.