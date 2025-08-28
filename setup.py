"""
Setup script for the Audio Analysis Pipeline package.
"""

from setuptools import setup, find_packages
from pathlib import Path


# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Audio Analysis Pipeline - Comprehensive audio processing and analysis toolkit"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []


setup(
    name="audio-analysis-pipeline",
    version="1.0.0",
    author="Audio Analysis Team",
    author_email="contact@audioanalysis.ai",
    description="Comprehensive audio analysis pipeline for speaker diarization, emotion recognition, and acoustic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-analysis-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/audio-analysis-pipeline/issues",
        "Documentation": "https://github.com/yourusername/audio-analysis-pipeline/wiki",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "output", "output.*"]),
    package_data={
        "config": ["*.yaml", "*.yml"],
        "src": ["*.json"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio-analysis=main:main",
            "audio-pipeline=main:cli",
        ],
    },
    zip_safe=False,
)