"""
Acoustic analysis module for extracting acoustic features from audio.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import librosa
import warnings

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    parselmouth = None
    call = None

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger, log_exception
from config.settings import ProcessingConfig
from src.models.segments import AcousticSegment
from src.models.base import BaseSegment


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)
warnings.filterwarnings('ignore', category=UserWarning)


    


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
        segments: Optional[List[Union[Dict, BaseSegment]]] = None,
        sample_rate: int = 16000
    ) -> List[AcousticSegment]:
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
                    # Handle both dict and BaseSegment types
                    if isinstance(segment, BaseSegment):
                        start_time = segment.start
                        end_time = segment.end
                    else:
                        start_time = segment['start']
                        end_time = segment['end']
                        
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        features = self.extract_features(
                            segment_audio,
                            sr,
                            start_time=start_time,
                            end_time=end_time
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
            
            return features_list
            
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
    ) -> AcousticSegment:
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
        
        # Create flattened acoustic features dictionary for the model
        flattened_features: Dict[str, float] = {}
        
        # Add energy features
        if rms_energy is not None:
            flattened_features['rms_energy'] = rms_energy
        if zcr is not None:
            flattened_features['zero_crossing_rate'] = zcr
        
        # Add spectral features
        for key, value in spectral_features.items():
            if value is not None:
                flattened_features[f'spectral_{key}'] = value
        
        # Add pitch features
        for key, value in pitch_features.items():
            if value is not None:
                flattened_features[f'pitch_{key}'] = value
        
        # Add formant features
        for key, value in formants.items():
            if value is not None:
                flattened_features[f'formant_{key}'] = value
        
        # Add voice quality features
        for key, value in voice_quality.items():
            if value is not None:
                flattened_features[f'voice_{key}'] = value
        
        # Add MFCC features (as separate entries)
        if mfcc is not None:
            for i, coeff in enumerate(mfcc):
                flattened_features[f'mfcc_{i}'] = float(coeff)
        
        # Add tempo
        if tempo is not None:
            flattened_features['tempo'] = tempo
        
        # Also store structured features for backward compatibility
        structured_features = {
            'energy': {
                'rms': rms_energy,
                'zero_crossing_rate': zcr
            },
            'spectral': spectral_features,
            'pitch': pitch_features,
            'formants': formants,
            'mfcc': mfcc.tolist() if mfcc is not None else None,
            'voice_quality': voice_quality,
            'tempo': tempo
        }
        
        # Create AcousticSegment with proper individual properties
        segment = AcousticSegment(
            start=start_time,
            end=end_time,
            confidence=1.0,  # Always confident in acoustic features
            pitch_mean=pitch_features.get('mean'),
            pitch_std=pitch_features.get('std'),
            rms_energy=rms_energy,
            spectral_centroid=spectral_features.get('centroid'),
            features=flattened_features
        )
        
        # Store structured features as additional attribute for backward compatibility
        segment.structured_features = structured_features  # type: ignore[attr-defined]
        
        return segment
    
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
            if PARSELMOUTH_AVAILABLE and parselmouth is not None:
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
            if PARSELMOUTH_AVAILABLE and parselmouth is not None:
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
            formants: Dict[str, Optional[float]] = {'f1': None, 'f2': None, 'f3': None}
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
            if PARSELMOUTH_AVAILABLE and parselmouth is not None and call is not None:
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
            f0, _, _ = librosa.pyin(
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
            # Use the newer API if available, fall back to deprecated one
            try:
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sample_rate)
            except AttributeError:
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
            
            if len(tempo) > 0:
                return float(tempo[0])
            
        except Exception as e:
            logger.debug(f"Failed to extract tempo: {e}")
        
        return None
    
    def get_feature_statistics(
        self,
        features_list: List[AcousticSegment]
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
        energy_values: List[float] = []
        zcr_values: List[float] = []
        centroid_values: List[float] = []
        pitch_values: List[float] = []
        tempo_values: List[float] = []
        
        for f in features_list:
            # Try to get values from individual properties first (preferred)
            if f.rms_energy is not None:
                energy_values.append(f.rms_energy)
            if hasattr(f, 'spectral_centroid') and f.spectral_centroid is not None:
                centroid_values.append(f.spectral_centroid)
            if f.pitch_mean is not None:
                pitch_values.append(f.pitch_mean)
                
            # Also check flattened features
            if f.features:
                if 'rms_energy' in f.features and f.rms_energy is None:
                    energy_values.append(f.features['rms_energy'])
                if 'zero_crossing_rate' in f.features:
                    zcr_values.append(f.features['zero_crossing_rate'])
                if 'spectral_centroid' in f.features and (not hasattr(f, 'spectral_centroid') or f.spectral_centroid is None):
                    centroid_values.append(f.features['spectral_centroid'])
                if 'pitch_mean' in f.features and f.pitch_mean is None:
                    pitch_values.append(f.features['pitch_mean'])
                if 'tempo' in f.features:
                    tempo_values.append(f.features['tempo'])
                    
            # Fallback to structured features if available
            if hasattr(f, 'structured_features'):
                struct_features = getattr(f, 'structured_features', None)
                if struct_features:
                    # Extract energy features
                    energy_dict = struct_features.get('energy', {})
                    if isinstance(energy_dict, dict):
                        if energy_dict.get('rms') is not None and len(energy_values) == 0:
                            energy_values.append(energy_dict['rms'])
                        if energy_dict.get('zero_crossing_rate') is not None:
                            zcr_values.append(energy_dict['zero_crossing_rate'])
                            
                    # Extract spectral features  
                    spectral_dict = struct_features.get('spectral', {})
                    if isinstance(spectral_dict, dict) and spectral_dict.get('centroid') is not None and len(centroid_values) == 0:
                        centroid_values.append(spectral_dict['centroid'])
                        
                    # Extract pitch features
                    pitch_dict = struct_features.get('pitch', {})
                    if isinstance(pitch_dict, dict) and pitch_dict.get('mean') is not None and len(pitch_values) == 0:
                        pitch_values.append(pitch_dict['mean'])
                        
                    # Extract tempo
                    if struct_features.get('tempo') is not None and len(tempo_values) == 0:
                        tempo_values.append(struct_features['tempo'])
        
        # Add statistics for each feature type
        feature_arrays = [
            (energy_values, 'energy'),
            (zcr_values, 'zcr'),
            (centroid_values, 'centroid'),
            (pitch_values, 'pitch'),
            (tempo_values, 'tempo')
        ]
        
        for values, display_name in feature_arrays:
            if values:
                feature_stats = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                stats[f'{display_name}_stats'] = feature_stats
        
        return stats