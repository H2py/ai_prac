"""
Speech Emotion Analysis - Main Analyzer Class
WhisperX + OpenSMILE 기반 화자별 감정 분석 시스템
"""

import whisperx
import opensmile
import torch
import librosa
import numpy as np
import pandas as pd
import os
import tempfile
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
from tqdm import tqdm

# 로컬 설정 import
from config import ModelConfig, ProcessingConfig, AnalysisConfig, LoggingConfig, Config
from utils.audio_utils import AudioProcessor
from utils.visualization import EmotionVisualizer

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning)

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, LoggingConfig.LOG_LEVEL),
    format=LoggingConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LoggingConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpeakerEmotionAnalyzer:
    """WhisperX + OpenSMILE 기반 화자별 감정 분석기"""
    
    def __init__(self, 
                 whisper_model: str = None,
                 emotion_classifier_path: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 device: Optional[str] = None):
        """
        분석기 초기화
        
        Args:
            whisper_model: WhisperX 모델 크기
            emotion_classifier_path: 사전 학습된 감정 분류기 경로
            hf_token: HuggingFace 토큰 (diarization용)
            device: 사용할 디바이스
        """
        
        # 설정 초기화
        self.whisper_model_size = whisper_model or ModelConfig.WHISPER_MODEL_SIZE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.hf_token = hf_token or ModelConfig.HUGGINGFACE_TOKEN
        
        logger.info(f"장치 초기화: {self.device}")
        logger.info(f"Whisper 모델: {self.whisper_model_size}")
        
        # 오디오 프로세서 초기화
        self.audio_processor = AudioProcessor()
        self.visualizer = EmotionVisualizer()
        
        # 모델들 초기화
        self._initialize_models(emotion_classifier_path)
        
        # 분석 통계
        self.analysis_stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'processing_time': 0.0
        }
    
    def _initialize_models(self, emotion_classifier_path: Optional[str]):
        """모든 필요한 모델들을 초기화합니다."""
        
        try:
            # WhisperX ASR 모델 초기화
            logger.info("WhisperX 모델 로딩 중...")
            self.asr_model = whisperx.load_model(
                self.whisper_model_size, 
                self.device, 
                compute_type=self.compute_type
            )
            logger.info("✅ WhisperX 모델 로딩 완료")
            
            # Speaker Diarization 모델 초기화
            if self.hf_token:
                logger.info("Speaker Diarization 모델 로딩 중...")
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, 
                    device=self.device
                )
                logger.info("✅ Speaker Diarization 모델 로딩 완료")
            else:
                logger.warning("HF 토큰이 없어 Speaker Diarization을 건너뜁니다.")
                self.diarize_model = None
                
            # OpenSMILE 초기화
            logger.info("OpenSMILE 초기화 중...")
            self.smile = opensmile.Smile(
                feature_set=getattr(opensmile.FeatureSet, ModelConfig.OPENSMILE_FEATURE_SET),
                feature_level=getattr(opensmile.FeatureLevel, ModelConfig.OPENSMILE_FEATURE_LEVEL),
            )
            logger.info("✅ OpenSMILE 초기화 완료")
            
            # 감정 분류기 로드 또는 초기화
            self._initialize_emotion_classifier(emotion_classifier_path)
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            raise
    
    def _initialize_emotion_classifier(self, classifier_path: Optional[str]):
        """감정 분류기를 초기화합니다."""
        
        if classifier_path and Path(classifier_path).exists():
            try:
                logger.info(f"감정 분류기 로딩: {classifier_path}")
                model_data = joblib.load(classifier_path)
                
                if isinstance(model_data, dict):
                    self.emotion_classifier = model_data.get('classifier')
                    self.scaler = model_data.get('scaler')
                else:
                    self.emotion_classifier = model_data
                    self.scaler = StandardScaler()
                
                self.is_trained = True
                logger.info("✅ 감정 분류기 로딩 완료")
                
            except Exception as e:
                logger.error(f"감정 분류기 로딩 실패: {e}")
                self._create_default_classifier()
        else:
            logger.info("사전 학습된 분류기가 없어 기본 분류기를 생성합니다.")
            self._create_default_classifier()
    
    def _create_default_classifier(self):
        """기본 감정 분류기를 생성합니다."""
        self.emotion_classifier = SVC(probability=True, random_state=42, kernel='rbf')
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def transcribe_and_diarize(self, audio_path: str) -> Dict[str, Any]:
        """음성 파일을 전사하고 화자를 분리합니다."""
        
        logger.info(f"오디오 처리 시작: {audio_path}")
        
        try:
            # 오디오 전처리
            if ProcessingConfig.NORMALIZE_AUDIO:
                audio_path = self.audio_processor.normalize_audio(audio_path)
            
            # 오디오 로드 및 전사
            audio = whisperx.load_audio(audio_path)
            result = self.asr_model.transcribe(
                audio, 
                batch_size=ModelConfig.WHISPER_BATCH_SIZE
            )
            
            logger.info(f"전사 완료: {len(result['segments'])}개 세그먼트")
            
            # Speaker diarization
            if self.diarize_model:
                logger.info("화자 분리 수행 중...")
                diarize_segments = self.diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("✅ 화자 분리 완료")
            else:
                # 단일 화자로 처리
                for segment in result['segments']:
                    segment['speaker'] = 'SPEAKER_00'
            
            return result
            
        except Exception as e:
            logger.error(f"전사/화자분리 실패: {e}")
            raise
    
    def extract_segment_audio(self, audio_path: str, start_time: float, end_time: float) -> str:
        """특정 시간 구간의 오디오를 추출하여 임시 파일로 저장합니다."""
        
        try:
            audio_data, sr = librosa.load(
                audio_path, 
                sr=ModelConfig.SAMPLE_RATE,
                offset=start_time,
                duration=end_time - start_time
            )
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            librosa.output.write_wav(temp_path, audio_data, sr)
            return temp_path
            
        except Exception as e:
            logger.error(f"오디오 세그먼트 추출 실패: {e}")
            raise
    
    def analyze_emotion_for_segment(self, audio_path: str, segment: Dict) -> Dict[str, Any]:
        """단일 세그먼트에 대한 감정 분석을 수행합니다."""
        
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        
        # 너무 짧은 세그먼트 건너뛰기
        if duration < ModelConfig.MIN_SEGMENT_DURATION:
            return self._create_empty_result(segment, "too_short")
        
        temp_audio_path = None
        try:
            # 세그먼트 오디오 추출
            temp_audio_path = self.extract_segment_audio(audio_path, start_time, end_time)
            
            # OpenSMILE로 특성 추출
            features = self.smile.process_file(temp_audio_path)
            
            if features.empty:
                return self._create_empty_result(segment, "no_features")
            
            # 감정 예측 (모델이 학습된 경우만)
            if self.is_trained:
                emotion_pred, confidence, probabilities = self._predict_emotion(features)
            else:
                emotion_pred = "unknown"
                confidence = 0.0
                probabilities = {}
            
            return {
                'speaker': segment.get('speaker', 'SPEAKER_00'),
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'text': segment.get('text', ''),
                'emotion': emotion_pred,
                'confidence': float(confidence),
                'emotion_probabilities': probabilities,
                'features': features.iloc[0].to_dict() if AnalysisConfig.SAVE_DETAILED_FEATURES else {},
                'quality_score': self._calculate_quality_score(features, confidence)
            }
            
        except Exception as e:
            logger.error(f"세그먼트 {start_time}-{end_time} 감정 분석 실패: {e}")
            return self._create_empty_result(segment, "error", str(e))
            
        finally:
            # 임시 파일 정리
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
    
    def _predict_emotion(self, features: pd.DataFrame) -> Tuple[str, float, Dict]:
        """특성으로부터 감정을 예측합니다."""
        
        try:
            # 특성 정규화
            features_scaled = self.scaler.transform(features.values.reshape(1, -1))
            
            # 예측 수행
            emotion_pred = self.emotion_classifier.predict(features_scaled)[0]
            emotion_proba = self.emotion_classifier.predict_proba(features_scaled)[0]
            
            # 확률 딕셔너리 생성
            probabilities = {
                label: float(prob) 
                for label, prob in zip(ModelConfig.EMOTION_LABELS, emotion_proba)
            }
            
            # 최대 확률 (신뢰도)
            confidence = float(np.max(emotion_proba))
            
            return emotion_pred, confidence, probabilities
            
        except Exception as e:
            logger.error(f"감정 예측 실패: {e}")
            return "error", 0.0, {}
    
    def _create_empty_result(self, segment: Dict, reason: str, error_msg: str = "") -> Dict[str, Any]:
        """빈 결과 객체를 생성합니다."""
        
        return {
            'speaker': segment.get('speaker', 'SPEAKER_00'),
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['end'] - segment['start'],
            'text': segment.get('text', ''),
            'emotion': reason,
            'confidence': 0.0,
            'emotion_probabilities': {},
            'features': {},
            'quality_score': 0.0,
            'error': error_msg if error_msg else None
        }
    
    def _calculate_quality_score(self, features: pd.DataFrame, confidence: float) -> float:
        """분석 품질 점수를 계산합니다."""
        
        try:
            # 기본 품질 점수는 신뢰도
            quality = confidence
            
            # 특성의 이상값 여부 확인 (간단한 Z-score 기반)
            feature_values = features.values.flatten()
            z_scores = np.abs((feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-8))
            outlier_ratio = np.mean(z_scores > 3.0)
            
            # 이상값이 많으면 품질 점수 감소
            quality *= (1.0 - outlier_ratio * 0.3)
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except:
            return confidence
    
    def analyze_audio(self, audio_path: str, save_results: bool = True) -> List[Dict[str, Any]]:
        """전체 오디오 파일에 대한 화자별 감정 분석을 수행합니다."""
        
        start_time = datetime.now()
        logger.info(f"감정 분석 시작: {audio_path}")
        
        try:
            # 1단계: 전사 및 화자 분리
            transcription_result = self.transcribe_and_diarize(audio_path)
            
            # 2단계: 각 세그먼트별 감정 분석
            emotion_results = []
            segments = transcription_result['segments']
            
            logger.info(f"{len(segments)}개 세그먼트의 감정 분석 시작")
            
            for i, segment in enumerate(tqdm(segments, desc="감정 분석 진행")):
                try:
                    emotion_result = self.analyze_emotion_for_segment(audio_path, segment)
                    emotion_results.append(emotion_result)
                    self.analysis_stats['successful_analyses'] += 1
                    
                except Exception as e:
                    logger.error(f"세그먼트 {i} 분석 실패: {e}")
                    self.analysis_stats['failed_analyses'] += 1
                    continue
            
            # 3단계: 결과 후처리
            emotion_results = self._post_process_results(emotion_results)
            
            # 4단계: 결과 저장
            if save_results:
                self._save_results(audio_path, emotion_results, transcription_result)
            
            # 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self.analysis_stats['total_processed'] += 1
            self.analysis_stats['processing_time'] += processing_time
            
            logger.info(f"✅ 분석 완료 ({processing_time:.2f}초): {len(emotion_results)}개 결과")
            
            return emotion_results
            
        except Exception as e:
            logger.error(f"오디오 분석 실패: {e}")
            self.analysis_stats['failed_analyses'] += 1
            raise
    
    def _post_process_results(self, results: List[Dict]) -> List[Dict]:
        """결과를 후처리합니다 (평활화, 이상값 제거 등)."""
        
        if not results:
            return results
        
        # 신뢰도가 너무 낮은 결과 필터링
        if AnalysisConfig.MIN_CONFIDENCE_THRESHOLD > 0:
            results = [
                r for r in results 
                if r['confidence'] >= AnalysisConfig.MIN_CONFIDENCE_THRESHOLD
            ]
        
        # 감정 레이블 정리
        for result in results:
            if result['emotion'] in ['too_short', 'no_features', 'error']:
                result['emotion'] = 'unknown'
        
        return results
    
    def _save_results(self, audio_path: str, emotion_results: List[Dict], transcription_result: Dict):
        """분석 결과를 저장합니다."""
        
        try:
            # 파일명 기반 결과 디렉토리 생성
            audio_name = Path(audio_path).stem
            result_dir = AnalysisConfig.EMOTIONS_PATH / audio_name
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 상세 감정 분석 결과 저장
            emotion_file = result_dir / "emotion_analysis.json"
            with open(emotion_file, 'w', encoding='utf-8') as f:
                json.dump(emotion_results, f, ensure_ascii=False, indent=2)
            
            # 전사 결과 저장
            transcription_file = result_dir / "transcription.json"
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_result, f, ensure_ascii=False, indent=2)
            
            # CSV 형태로도 저장
            if "csv" in AnalysisConfig.EXPORT_FORMATS:
                df = pd.DataFrame(emotion_results)
                df.to_csv(result_dir / "emotion_analysis.csv", index=False)
            
            # 요약 보고서 생성
            if AnalysisConfig.GENERATE_SUMMARY_REPORT:
                summary = self.generate_analysis_summary(emotion_results)
                summary_file = result_dir / "summary_report.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # 시각화
            if AnalysisConfig.PLOT_EMOTION_TIMELINE:
                self.visualizer.plot_emotion_timeline(
                    emotion_results, 
                    save_path=result_dir / "emotion_timeline.png"
                )
            
            logger.info(f"결과 저장 완료: {result_dir}")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    def train_emotion_classifier(self, 
                                 train_audio_paths: List[str], 
                                 train_labels: List[str],
                                 test_size: float = 0.2,
                                 classifier_type: str = "svm",
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """감정 분류기를 학습합니다."""
        
        logger.info(f"{len(train_audio_paths)}개 샘플로 감정 분류기 학습 시작")
        
        try:
            # 특성 추출
            logger.info("특성 추출 중...")
            all_features = []
            valid_labels = []
            
            for i, (audio_path, label) in enumerate(tqdm(
                zip(train_audio_paths, train_labels), 
                desc="특성 추출",
                total=len(train_audio_paths)
            )):
                try:
                    features = self.smile.process_file(audio_path)
                    if not features.empty:
                        all_features.append(features.iloc[0].values)
                        valid_labels.append(label)
                except Exception as e:
                    logger.warning(f"파일 {audio_path} 특성 추출 실패: {e}")
                    continue
            
            if len(all_features) == 0:
                raise ValueError("유효한 특성을 추출할 수 없습니다.")
            
            # 데이터 준비
            X = np.array(all_features)
            y = np.array(valid_labels)
            
            logger.info(f"유효한 샘플 수: {len(X)}")
            
            # 학습/검증 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # 데이터 정규화
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 분류기 선택 및 학습
            if classifier_type.lower() == "svm":
                self.emotion_classifier = SVC(probability=True, random_state=42, kernel='rbf')
            elif classifier_type.lower() == "rf":
                self.emotion_classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
            else:
                raise ValueError(f"지원하지 않는 분류기: {classifier_type}")
            
            logger.info("모델 학습 중...")
            self.emotion_classifier.fit(X_train_scaled, y_train)
            
            # 성능 평가
            y_pred = self.emotion_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # 학습 완료 플래그 설정
            self.is_trained = True
            ModelConfig.EMOTION_LABELS = list(set(valid_labels))
            
            # 결과 정리
            training_results = {
                'accuracy': accuracy,
                'classification_report': classification_rep,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': X.shape[1],
                'emotion_labels': ModelConfig.EMOTION_LABELS,
                'classifier_type': classifier_type,
                'training_date': datetime.now().isoformat()
            }
            
            # 모델 저장
            save_path = save_path or (Config.MODELS_DIR / "emotion_classifier.pkl")
            model_data = {
                'classifier': self.emotion_classifier,
                'scaler': self.scaler,
                'emotion_labels': ModelConfig.EMOTION_LABELS,
                'training_results': training_results
            }
            joblib.dump(model_data, save_path)
            logger.info(f"모델 저장 완료: {save_path}")
            
            logger.info(f"✅ 학습 완료! 정확도: {accuracy:.3f}")
            return training_results
            
        except Exception as e:
            logger.error(f"분류기 학습 실패: {e}")
            raise
    
    def generate_analysis_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """분석 결과의 요약을 생성합니다."""
        
        if not results:
            return {'error': 'No results to summarize'}
        
        try:
            df = pd.DataFrame(results)
            
            summary = {
                'overview': {
                    'total_segments': len(results),
                    'total_duration': float(df['duration'].sum()),
                    'unique_speakers': len(df['speaker'].unique()),
                    'avg_confidence': float(df['confidence'].mean()),
                    'analysis_date': datetime.now().isoformat()
                },
                'speaker_analysis': {},
                'emotion_distribution': {},
                'quality_metrics': {
                    'high_confidence_segments': int((df['confidence'] > AnalysisConfig.HIGH_CONFIDENCE_THRESHOLD).sum()),
                    'low_confidence_segments': int((df['confidence'] < AnalysisConfig.MIN_CONFIDENCE_THRESHOLD).sum()),
                    'avg_quality_score': float(df['quality_score'].mean()) if 'quality_score' in df else 0.0
                }
            }
            
            # 화자별 분석
            for speaker in df['speaker'].unique():
                speaker_data = df[df['speaker'] == speaker]
                summary['speaker_analysis'][speaker] = {
                    'segment_count': len(speaker_data),
                    'total_duration': float(speaker_data['duration'].sum()),
                    'dominant_emotion': speaker_data['emotion'].mode().iloc[0] if not speaker_data['emotion'].empty else 'unknown',
                    'avg_confidence': float(speaker_data['confidence'].mean()),
                    'emotion_distribution': speaker_data['emotion'].value_counts().to_dict()
                }
            
            # 전체 감정 분포
            summary['emotion_distribution'] = df['emotion'].value_counts().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """분석기의 통계 정보를 반환합니다."""
        
        stats = self.analysis_stats.copy()
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / (stats['successful_analyses'] + stats['failed_analyses'])
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def cleanup(self):
        """리소스를 정리합니다."""
        
        try:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 실패: {e}")

if __name__ == "__main__":
    # 간단한 테스트
    print("SpeakerEmotionAnalyzer 테스트")
    
    analyzer = SpeakerEmotionAnalyzer()
    print(f"초기화 완료 - 학습 상태: {analyzer.is_trained}")
    print(f"통계: {analyzer.get_analysis_statistics()}")
    
    analyzer.cleanup()