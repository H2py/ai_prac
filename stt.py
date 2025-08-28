import torch
import time
import os
from pyannote.audio import Model, Inference
import speech_recognition as sr
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class WhisperSTT:
    def __init__(self):
        # GPU 사용 가능하면 cuda, 없으면 cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # PyAnnote embedding 모델 로드
        try:
            self.embeddingmodel = Model.from_pretrained(
                checkpoint="pyannote/embedding", 
                cache_dir="models/pyannote",
                use_auth_token=os.getenv("HF_API_KEY")  # Hugging Face API 키
            )
            
            # Inference 객체 생성 (화자 임베딩 추출용)
            self.inference = Inference(
                self.embeddingmodel, 
                window="whole", 
                device=self.device
            )
            
            # 메인 화자 임베딩 (기준이 되는 화자)
            self.main_speaker_embedding = self.inference("austin.wav")
            print("✅ PyAnnote embedding model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ PyAnnote model loading failed: {e}")
            self.embeddingmodel = None
            self.inference = None
            self.main_speaker_embedding = None
        
        # Speech Recognition 초기화
        self.recognizer = sr.Recognizer()
        
    def callback(self, recognizer, audio):
        """실시간 음성 인식 콜백 함수"""
        try:
            print("🎤 Processing audio data...")
            
            # Google Speech Recognition 사용
            # 실제 사용시에는 Google API 키가 필요합니다
            # 테스트 목적으로는 기본 API 키 사용
            
            # 오디오 데이터를 WAV 파일로 저장
            transcribed_file = "transcribed-audio.wav"
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())
            
            # 처리 시간 측정 시작
            start = time.time()
            
            # Google Speech Recognition으로 음성 인식
            output = recognizer.recognize_google(audio, language="ko-KR")
            
            # 처리 시간 측정 종료
            end = time.time()
            
            print(f"🎧 Listen Offline Transcription in {end - start:.2f} seconds: {output}")
            
            # 텍스트 후처리 (특수문자 제거, 소문자 변환)
            clean_output = self.clean_text(output)
            
            # 화자 임베딩 분석 (PyAnnote 모델이 로드된 경우)
            if self.inference and self.main_speaker_embedding is not None:
                self.analyze_speaker(transcribed_file)
            
            return clean_output
            
        except sr.UnknownValueError:
            print("❌ Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            print(f"❌ Error in callback: {e}")
            return None
    
    def clean_text(self, text):
        """텍스트 후처리"""
        if not text:
            return ""
        
        # 특수문자 제거 및 정규화
        clean_output = str(text).replace(".", "").replace(",", "").replace("?", "").replace("!", "").lower()
        print(f"🧹 Cleaned text: {clean_output}")
        
        return clean_output
    
    def analyze_speaker(self, audio_file):
        """화자 분석 (PyAnnote embedding 사용)"""
        try:
            # 현재 오디오의 화자 임베딩 추출
            current_embedding = self.inference(audio_file)
            
            # 메인 화자와의 유사도 계산 (코사인 유사도)
            similarity = torch.cosine_similarity(
                self.main_speaker_embedding.unsqueeze(0), 
                current_embedding.unsqueeze(0)
            )
            
            print(f"🎭 Speaker similarity: {similarity.item():.3f}")
            
            # 유사도가 0.8 이상이면 같은 화자로 판단
            if similarity.item() > 0.8:
                print("✅ Same speaker detected")
            else:
                print("🆕 Different speaker detected")
                
        except Exception as e:
            print(f"⚠️ Speaker analysis failed: {e}")
    
    def start_listening(self):
        """실시간 음성 인식 시작"""
        print("🎤 Starting real-time speech recognition...")
        print("💡 Speak into your microphone. Press Ctrl+C to stop.")
        
        with sr.Microphone() as source:
            # 환경 소음 조정
            print("🔧 Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Ready! Start speaking...")
        
        # 백그라운드에서 실시간 음성 인식 시작
        stop_listening = self.recognizer.listen_in_background(
            sr.Microphone(), 
            self.callback,
            phrase_time_limit=5  # 5초마다 처리
        )
        
        try:
            while True:
                time.sleep(0.1)  # CPU 사용률 낮추기
        except KeyboardInterrupt:
            print("\n🛑 Stopping speech recognition...")
            stop_listening(wait_for_stop=False)
    
    def transcribe_file(self, audio_file_path):
        """파일에서 음성 인식"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            print(f"🎵 Transcribing file: {audio_file_path}")
            start = time.time()
            
            # Google Speech Recognition
            result = self.recognizer.recognize_google(audio, language="ko-KR")
            
            end = time.time()
            print(f"📝 Transcription completed in {end - start:.2f} seconds")
            print(f"📄 Result: {result}")
            
            # 화자 분석
            if self.inference:
                self.analyze_speaker(audio_file_path)
            
            return self.clean_text(result)
            
        except Exception as e:
            print(f"❌ File transcription failed: {e}")
            return None

# 사용 예시
if __name__ == "__main__":
    # WhisperSTT 인스턴스 생성
    stt = WhisperSTT()
    
    print("=" * 50)
    print("🎤 WhisperSTT + PyAnnote 음성 인식 시스템")
    print("=" * 50)
    print("1. 실시간 음성 인식")
    print("2. 오디오 파일 분석")
    print("=" * 50)
    
    choice = input("선택하세요 (1 또는 2): ")
    
    if choice == "1":
        # 실시간 음성 인식 시작
        stt.start_listening()
    
    elif choice == "2":
        # 파일 경로 입력
        file_path = input("오디오 파일 경로를 입력하세요: ")
        if os.path.exists(file_path):
            result = stt.transcribe_file(file_path)
            if result:
                print(f"\n✅ 최종 결과: {result}")
        else:
            print("❌ 파일을 찾을 수 없습니다.")
    
    else:
        print("❌ 잘못된 선택입니다.")

"""
필요한 설정:

1. .env 파일에 추가:
HF_API_KEY=your_huggingface_api_token

2. 의존성 설치:
pip install torch
pip install pyannote.audio
pip install speechrecognition
pip install pyaudio

3. Hugging Face 토큰 생성:
- https://huggingface.co/settings/tokens 에서 토큰 생성
- pyannote 모델 사용 권한 승인 필요

4. 기준 화자 오디오 파일:
- austin.wav 파일을 프로젝트 루트에 배치
- 또는 코드에서 다른 파일명으로 수정
"""