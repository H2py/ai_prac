# scripts/check_installation.py

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}: 설치됨")
        return True
    except ImportError:
        print(f"❌ {package_name}: 설치되지 않음")
        return False

packages = [
    ("PyTorch", "torch"),
    ("Torchaudio", "torchaudio"), 
    ("WhisperX", "whisperx"),
    ("OpenSMILE", "opensmile"),
    ("Librosa", "librosa"),
    ("Scikit-learn", "sklearn"),
    ("Pandas", "pandas"),
    ("NumPy", "numpy"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Pyannote Audio", "pyannote.audio"),
]

print("패키지 설치 확인 중...")
print("=" * 40)

all_installed = True
for package_name, import_name in packages:
    if not check_package(package_name, import_name):
        all_installed = False

print("=" * 40)
if all_installed:
    print("🎉 모든 패키지가 성공적으로 설치되었습니다!")
else:
    print("⚠️  일부 패키지 설치에 문제가 있습니다.")

# CUDA 확인
try:
    import torch
    if torch.cuda.is_available():
        print(f"🚀 CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 CPU 모드로 실행됩니다.")
except:
    print("❌ PyTorch CUDA 확인 실패")