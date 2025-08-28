"""
Acoustic analysis module for extracting acoustic features from audio.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import librosa
from dataclasses import dataclass
import warnings

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    parselmouth = None

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger, log_exception
from config.settings import ProcessingConfig


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class AcousticFeatures:
    """Container for acoustic features."""
    
    start: float
    end: float
    
    # Energy-based features
    rms_energy: float
    zero_crossing_rate: float
    
    # Spectral features
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flux: float
    spectral_bandwidth: float
    
    # Pitch features
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    pitch_min: Optional[float] = None
    pitch_max: Optional[float] = None
    
    # Formants
    f1: Optional[float] = None
    f2: Optional[float] = None
    f3: Optional[float] = None
    
    # MFCC features (13 coefficients)
    mfcc: Optional[np.ndarray] = None
    
    # Voice quality
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    hnr: Optional[float] = None  # Harmonics-to-noise ratio
    
    # Tempo
    tempo: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'start': self.start,
            'end': self.end,
            'duration': self.end - self.start,
            'energy': {
                'rms': self.rms_energy,
                'zero_crossing_rate': self.zero_crossing_rate
            },
            'spectral': {
                'centroid': self.spectral_centroid,
                'rolloff': self.spectral_rolloff,
                'flux': self.spectral_flux,
                'bandwidth': self.spectral_bandwidth
            }
        }
        
        # Add pitch features if available
        if self.pitch_mean is not None:
            result['pitch'] = {
                'mean': self.pitch_mean,
                'std': self.pitch_std,
                'min': self.pitch_min,
                'max': self.pitch_max,
                'range': self.pitch_max - self.pitch_min if self.pitch_max and self.pitch_min else 0
            }
        
        # Add formants if available
        if self.f1 is not None:
            result['formants'] = {
                'f1': self.f1,
                'f2': self.f2,
                'f3': self.f3
            }
        
        # Add MFCC if available
        if self.mfcc is not None:
            result['mfcc'] = self.mfcc.tolist() if isinstance(self.mfcc, np.ndarray) else self.mfcc
        
        # Add voice quality if available
        if self.jitter is not None:
            result['voice_quality'] = {
                'jitter': self.jitter,
                'shimmer': self.shimmer,
                'hnr': self.hnr
            }
        
        # Add tempo if available
        if self.tempo is not None:
            result['tempo'] = self.tempo
        
        return result


class AcousticAnalyzer:
    """Extract acoustic features from audio segments."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize acoustic analyzer.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.sample_rate = 16000  # Default sample rate
        
        # Check if parselmouth is available
        if PARSELMOUTH_AVAILABLE:
            logger.info("Parselmouth available for advanced acoustic analysis")
        else:
            logger.warning("Parselmouth not available, using basic acoustic features only")
    
    def analyze(
        self,
        audio_path: Union[str, Path],
        segments: Optional[List[Dict]] = None,
        sample_rate: int = 16000
    ) -> List[AcousticFeatures]:
        """Extract acoustic features for audio segments.
        
        Args:
            audio_path: Path to audio file
            segments: Optional segments to analyze
            sample_rate: Sample rate
            
        Returns:
            List of acoustic features
        """
        perf_logger.start_timer("acoustic_analysis")
        audio_path = Path(audio_path)
        self.sample_rate = sample_rate
        
        try:
            logger.info(f"Analyzing acoustic features for: {audio_path}")
            
            # Load audio
            audio_data, sr = load_audio(audio_path, sample_rate=sample_rate)
            
            features_list = []
            
            if segments:
                # Analyze provided segments
                progress = ProgressLogger(logger, total=len(segments))
                
                for segment in segments:
                    start_sample = int(segment['start'] * sr)
                    end_sample = int(segment['end'] * sr)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        features = self.extract_features(
                            segment_audio,
                            sr,
                            start_time=segment['start'],
                            end_time=segment['end']
                        )
                        features_list.append(features)
                    
                    progress.update(message=f"Processed segment {len(features_list)}")
                
                progress.complete()
                
            else:
                # Analyze whole audio in chunks
                chunks = split_audio_chunks(
                    audio_data,
                    sr,
                    chunk_duration=30.0,
                    overlap=2.0
                )
                
                progress = ProgressLogger(logger, total=len(chunks))
                
                for chunk, (start_time, end_time) in chunks:
                    if len(chunk) > 0:
                        features = self.extract_features(
                            chunk,
                            sr,
                            start_time=start_time,
                            end_time=end_time
                        )
                        features_list.append(features)
                    
                    progress.update(message=f"Processed chunk {len(features_list)}")
                
                progress.complete()
            
            duration = perf_logger.stop_timer("acoustic_analysis")
            logger.info(f"Acoustic analysis completed in {duration:.2f}s")
            
            # Convert dataclasses to dictionaries for compatibility
            feature_dicts = []
            for feat in features_list:
                feature_dicts.append(feat.to_dict())
            
            return feature_dicts
            
        except Exception as e:
            perf_logger.stop_timer("acoustic_analysis")
            log_exception(logger, e, "Acoustic analysis failed")
            return []
    
    def extract_features(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> AcousticFeatures:
        """Extract all acoustic features from audio chunk.
        
        Args:
            audio_chunk: Audio data array
            sample_rate: Sample rate
            start_time: Start time of chunk
            end_time: End time of chunk
            
        Returns:
            Acoustic features
        """
        if end_time is None:
            end_time = start_time + len(audio_chunk) / sample_rate
        
        # Extract energy-based features
        rms_energy = self._extract_rms_energy(audio_chunk)
        zcr = self._extract_zero_crossing_rate(audio_chunk)
        
        # Extract spectral features
        spectral_features = self._extract_spectral_features(audio_chunk, sample_rate)
        
        # Extract pitch features
        pitch_features = self._extract_pitch_features(audio_chunk, sample_rate)
        
        # Extract formants
        formants = self._extract_formants(audio_chunk, sample_rate)
        
        # Extract MFCCs
        mfcc = self._extract_mfcc(audio_chunk, sample_rate)
        
        # Extract voice quality
        voice_quality = self._extract_voice_quality(audio_chunk, sample_rate)
        
        # Extract tempo
        tempo = self._extract_tempo(audio_chunk, sample_rate)
        
        return AcousticFeatures(
            start=start_time,
            end=end_time,
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_features['centroid'],
            spectral_rolloff=spectral_features['rolloff'],
            spectral_flux=spectral_features['flux'],
            spectral_bandwidth=spectral_features['bandwidth'],
            pitch_mean=pitch_features.get('mean'),
            pitch_std=pitch_features.get('std'),
            pitch_min=pitch_features.get('min'),
            pitch_max=pitch_features.get('max'),
            f1=formants.get('f1'),
            f2=formants.get('f2'),
            f3=formants.get('f3'),
            mfcc=mfcc,
            jitter=voice_quality.get('jitter'),
            shimmer=voice_quality.get('shimmer'),
            hnr=voice_quality.get('hnr'),
            tempo=tempo
        )
    
    def _extract_rms_energy(self, audio_chunk: np.ndarray) -> float:
        """Extract RMS energy.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            RMS energy value
        """
        return float(np.sqrt(np.mean(audio_chunk ** 2)))
    
    def _extract_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """Extract zero crossing rate.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            Zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0]
        return float(np.mean(zcr))
    
    def _extract_spectral_features(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Extract spectral features.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        try:
            # Compute spectrogram
            stft = librosa.stft(audio_chunk, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio_chunk,
                sr=sample_rate
            )[0]
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_chunk,
                sr=sample_rate,
                roll_percent=0.85
            )[0]
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_chunk,
                sr=sample_rate
            )[0]
            
            # Spectral flux (simplified)
            flux = np.mean(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))
            
            return {
                'centroid': float(np.mean(centroid)),
                'rolloff': float(np.mean(rolloff)),
                'bandwidth': float(np.mean(bandwidth)),
                'flux': float(flux)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract spectral features: {e}")
            return {
                'centroid': 0.0,
                'rolloff': 0.0,
                'bandwidth': 0.0,
                'flux': 0.0
            }
    
    def _extract_pitch_features(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Optional[float]]:
        """Extract pitch (F0) features.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of pitch features
        """
        try:
            if PARSELMOUTH_AVAILABLE:
                # Use Parselmouth for accurate pitch extraction
                sound = parselmouth.Sound(audio_chunk, sampling_frequency=sample_rate)
                pitch = sound.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
                
                if len(pitch_values) > 0:
                    return {
                        'mean': float(np.mean(pitch_values)),
                        'std': float(np.std(pitch_values)),
                        'min': float(np.min(pitch_values)),
                        'max': float(np.max(pitch_values))
                    }
            else:
                # Use librosa's pitch detection
                pitches, magnitudes = librosa.piptrack(
                    y=audio_chunk,
                    sr=sample_rate,
                    fmin=50,
                    fmax=500
                )
                
                # Extract pitch values
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    return {
                        'mean': float(np.mean(pitch_values)),
                        'std': float(np.std(pitch_values)),
                        'min': float(np.min(pitch_values)),
                        'max': float(np.max(pitch_values))
                    }
            
        except Exception as e:
            logger.debug(f"Failed to extract pitch features: {e}")
        
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    
    def _extract_formants(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Optional[float]]:
        """Extract formant frequencies.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of formant frequencies
        """
        try:
            if PARSELMOUTH_AVAILABLE:
                # Use Parselmouth for formant extraction
                sound = parselmouth.Sound(audio_chunk, sampling_frequency=sample_rate)
                formant = sound.to_formant_burg()
                
                # Get mean formant values
                f1_values = []
                f2_values = []
                f3_values = []
                
                for i in range(int(formant.get_number_of_frames())):
                    time = formant.get_time_from_frame_number(i)
                    f1 = formant.get_value_at_time(1, time)
                    f2 = formant.get_value_at_time(2, time)
                    f3 = formant.get_value_at_time(3, time)
                    
                    if f1 and not np.isnan(f1):
                        f1_values.append(f1)
                    if f2 and not np.isnan(f2):
                        f2_values.append(f2)
                    if f3 and not np.isnan(f3):
                        f3_values.append(f3)
                
                return {
                    'f1': float(np.mean(f1_values)) if f1_values else None,
                    'f2': float(np.mean(f2_values)) if f2_values else None,
                    'f3': float(np.mean(f3_values)) if f3_values else None
                }
            else:
                # Simplified formant estimation using LPC
                # This is a rough approximation
                return self._estimate_formants_lpc(audio_chunk, sample_rate)
            
        except Exception as e:
            logger.debug(f"Failed to extract formants: {e}")
        
        return {'f1': None, 'f2': None, 'f3': None}
    
    def _estimate_formants_lpc(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Optional[float]]:
        """Estimate formants using LPC (simplified).
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of estimated formant frequencies
        """
        try:
            # Apply pre-emphasis
            pre_emphasized = np.append(audio_chunk[0], audio_chunk[1:] - 0.97 * audio_chunk[:-1])
            
            # LPC analysis
            order = 2 + sample_rate // 1000
            lpc = librosa.lpc(pre_emphasized, order=order)
            
            # Find roots and convert to frequencies
            roots = np.roots(lpc)
            roots = roots[np.imag(roots) >= 0]
            
            angles = np.angle(roots)
            frequencies = angles * (sample_rate / (2 * np.pi))
            frequencies = frequencies[frequencies > 90]
            frequencies = np.sort(frequencies)
            
            # Approximate formants
            formants = {'f1': None, 'f2': None, 'f3': None}
            if len(frequencies) >= 1:
                formants['f1'] = float(frequencies[0])
            if len(frequencies) >= 2:
                formants['f2'] = float(frequencies[1])
            if len(frequencies) >= 3:
                formants['f3'] = float(frequencies[2])
            
            return formants
            
        except Exception:
            return {'f1': None, 'f2': None, 'f3': None}
    
    def _extract_mfcc(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        n_mfcc: int = 13
    ) -> Optional[np.ndarray]:
        """Extract MFCC features.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC feature vector
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_chunk,
                sr=sample_rate,
                n_mfcc=n_mfcc
            )
            # Return mean of MFCCs over time
            return np.mean(mfcc, axis=1)
            
        except Exception as e:
            logger.debug(f"Failed to extract MFCC: {e}")
            return None
    
    def _extract_voice_quality(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Optional[float]]:
        """Extract voice quality measures.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of voice quality measures
        """
        try:
            if PARSELMOUTH_AVAILABLE:
                sound = parselmouth.Sound(audio_chunk, sampling_frequency=sample_rate)
                
                # Extract jitter
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
                jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                
                # Extract shimmer
                shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                
                # Extract harmonics-to-noise ratio
                hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr_value = call(hnr, "Get mean", 0, 0)
                
                return {
                    'jitter': float(jitter) if not np.isnan(jitter) else None,
                    'shimmer': float(shimmer) if not np.isnan(shimmer) else None,
                    'hnr': float(hnr_value) if not np.isnan(hnr_value) else None
                }
            else:
                # Basic approximation without Parselmouth
                return self._estimate_voice_quality(audio_chunk, sample_rate)
            
        except Exception as e:
            logger.debug(f"Failed to extract voice quality: {e}")
        
        return {'jitter': None, 'shimmer': None, 'hnr': None}
    
    def _estimate_voice_quality(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Optional[float]]:
        """Estimate voice quality without Parselmouth.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Estimated voice quality measures
        """
        try:
            # Simple energy-based estimations
            # These are rough approximations
            
            # Estimate jitter (pitch variation)
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk,
                fmin=50,
                fmax=500,
                sr=sample_rate
            )
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 1:
                jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean)
            else:
                jitter = None
            
            # Estimate shimmer (amplitude variation)
            amplitude = np.abs(audio_chunk)
            peaks = librosa.util.peak_pick(amplitude, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0, wait=10)
            if len(peaks) > 1:
                peak_values = amplitude[peaks]
                shimmer = np.std(peak_values) / np.mean(peak_values)
            else:
                shimmer = None
            
            # Estimate HNR (simplified)
            # This is a very rough approximation
            autocorr = np.correlate(audio_chunk, audio_chunk, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            
            if len(autocorr) > sample_rate // 50:  # At least 20ms
                # Find the peak in the expected pitch range
                min_lag = sample_rate // 500  # 500 Hz max
                max_lag = sample_rate // 50   # 50 Hz min
                
                if max_lag < len(autocorr):
                    pitch_peak = np.max(autocorr[min_lag:max_lag])
                    noise_floor = np.mean(autocorr[max_lag:])
                    
                    if noise_floor > 0:
                        hnr = 10 * np.log10(pitch_peak / noise_floor)
                    else:
                        hnr = None
                else:
                    hnr = None
            else:
                hnr = None
            
            return {
                'jitter': float(jitter) if jitter is not None else None,
                'shimmer': float(shimmer) if shimmer is not None else None,
                'hnr': float(hnr) if hnr is not None else None
            }
            
        except Exception:
            return {'jitter': None, 'shimmer': None, 'hnr': None}
    
    def _extract_tempo(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Optional[float]:
        """Extract tempo (BPM).
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Tempo in BPM
        """
        try:
            # Extract tempo using onset detection
            onset_env = librosa.onset.onset_strength(y=audio_chunk, sr=sample_rate)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
            
            if len(tempo) > 0:
                return float(tempo[0])
            
        except Exception as e:
            logger.debug(f"Failed to extract tempo: {e}")
        
        return None
    
    def get_feature_statistics(
        self,
        features_list: List[AcousticFeatures]
    ) -> Dict[str, Any]:
        """Calculate statistics from acoustic features.
        
        Args:
            features_list: List of acoustic features
            
        Returns:
            Dictionary with feature statistics
        """
        if not features_list:
            return {}
        
        # Collect all features
        stats = {
            'total_segments': len(features_list),
            'total_duration': sum(f.end - f.start for f in features_list)
        }
        
        # Calculate statistics for each feature type
        feature_types = [
            ('rms_energy', 'energy'),
            ('zero_crossing_rate', 'zcr'),
            ('spectral_centroid', 'centroid'),
            ('pitch_mean', 'pitch'),
            ('tempo', 'tempo')
        ]
        
        for attr_name, display_name in feature_types:
            values = [getattr(f, attr_name) for f in features_list 
                     if getattr(f, attr_name) is not None]
            
            if values:
                stats[f'{display_name}_stats'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return stats