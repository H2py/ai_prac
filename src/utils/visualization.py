"""
Visualization utilities for audio analysis results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot audio waveform.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(len(audio_data)) / sample_rate
    ax.plot(time, audio_data, linewidth=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Waveform plot saved to {save_path}")
    
    return fig


def plot_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    hop_length: int = 512,
    n_fft: int = 2048,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot spectrogram of audio.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        hop_length: Hop length for STFT
        n_fft: FFT size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    import librosa
    import librosa.display
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Display
    img = librosa.display.specshow(
        D,
        y_axis='linear',
        x_axis='time',
        sr=sample_rate,
        hop_length=hop_length,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    # Colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Spectrogram saved to {save_path}")
    
    return fig


def plot_speaker_timeline(
    segments: List[Dict[str, Any]],
    title: str = "Speaker Timeline",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot speaker diarization timeline.
    
    Args:
        segments: List of speaker segments
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique speakers and assign colors
    speakers = list(set(seg['speaker'] for seg in segments))
    colors = sns.color_palette("husl", len(speakers))
    speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}
    
    # Plot segments
    for i, segment in enumerate(segments):
        start = segment['start']
        duration = segment['end'] - start
        speaker = segment['speaker']
        
        rect = patches.Rectangle(
            (start, speakers.index(speaker)),
            duration,
            0.8,
            facecolor=speaker_colors[speaker],
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)
    
    # Set labels and limits
    ax.set_xlim(0, max(seg['end'] for seg in segments))
    ax.set_ylim(-0.5, len(speakers) - 0.5)
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speaker')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Speaker timeline saved to {save_path}")
    
    return fig


def plot_emotion_timeline(
    predictions: List[Dict[str, Any]],
    title: str = "Emotion Timeline",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot emotion predictions over time.
    
    Args:
        predictions: List of emotion predictions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Get emotion labels and colors
    emotion_labels = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
    emotion_colors = {
        'happy': '#FFD700',
        'sad': '#4169E1',
        'angry': '#DC143C',
        'fear': '#8B008B',
        'disgust': '#228B22',
        'surprise': '#FF69B4',
        'neutral': '#808080'
    }
    
    # Plot primary emotions as timeline
    for pred in predictions:
        start = pred.get('start', 0)
        end = pred.get('end', start + 1)
        emotion = pred.get('primary_emotion', 'neutral')
        confidence = pred.get('confidence', 1.0)
        
        color = emotion_colors.get(emotion, '#808080')
        alpha = 0.3 + 0.7 * confidence  # Scale alpha by confidence
        
        rect = patches.Rectangle(
            (start, 0),
            end - start,
            1,
            facecolor=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=0.5
        )
        ax1.add_patch(rect)
        
        # Add text label if segment is wide enough
        if (end - start) > 0.5:
            ax1.text(
                (start + end) / 2,
                0.5,
                f"{emotion}\n{confidence:.0%}",
                ha='center',
                va='center',
                fontsize=8
            )
    
    ax1.set_xlim(0, max(pred.get('end', 0) for pred in predictions) if predictions else 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time (s)')
    ax1.set_title(title)
    ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot confidence scores
    times = [(p.get('start', 0) + p.get('end', 0)) / 2 for p in predictions]
    confidences = [p.get('confidence', 0) for p in predictions]
    
    ax2.plot(times, confidences, 'o-', markersize=4, linewidth=1)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Emotion Detection Confidence')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    
    # Add legend for emotions
    handles = [patches.Patch(color=color, label=emotion) 
              for emotion, color in emotion_colors.items()]
    ax1.legend(handles=handles, loc='upper right', ncol=len(emotion_labels), 
              framealpha=0.8, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Emotion timeline saved to {save_path}")
    
    return fig


def plot_acoustic_features(
    features: List[Dict[str, Any]],
    feature_names: List[str] = None,
    title: str = "Acoustic Features",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot acoustic features over time.
    
    Args:
        features: List of acoustic feature dictionaries
        feature_names: List of feature names to plot
        title: Plot title  
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if not features:
        logger.warning("No features to plot")
        return None
    
    # Default features to plot
    if feature_names is None:
        feature_names = ['rms_energy', 'spectral_centroid', 'pitch_mean', 'zero_crossing_rate']
    
    # Filter available features
    available_features = []
    for name in feature_names:
        if any(name in f for f in features):
            available_features.append(name)
    
    if not available_features:
        logger.warning("No available features to plot")
        return None
    
    n_features = len(available_features)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature_name in enumerate(available_features):
        ax = axes[idx]
        
        # Extract feature values and times
        times = []
        values = []
        
        for f in features:
            if feature_name in f and f[feature_name] is not None:
                time = (f.get('start', 0) + f.get('end', 0)) / 2
                times.append(time)
                values.append(f[feature_name])
        
        if times and values:
            ax.plot(times, values, 'o-', markersize=3, linewidth=1)
            ax.set_ylabel(feature_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(times) > 2:
                z = np.polyfit(times, values, 1)
                p = np.poly1d(z)
                ax.plot(times, p(times), 'r--', alpha=0.5, linewidth=1)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Acoustic features plot saved to {save_path}")
    
    return fig


def plot_summary_statistics(
    results: Dict[str, Any],
    title: str = "Analysis Summary",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot summary statistics from analysis results.
    
    Args:
        results: Analysis results dictionary
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Speaker distribution (if available)
    if 'summary' in results and 'speaker_distribution' in results['summary']:
        ax1 = fig.add_subplot(gs[0, 0])
        speakers = list(results['summary']['speaker_distribution'].keys())
        percentages = list(results['summary']['speaker_distribution'].values())
        
        colors = sns.color_palette("husl", len(speakers))
        ax1.pie(percentages, labels=speakers, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Speaker Distribution')
    
    # 2. Emotion distribution (if available)
    if 'emotions' in results and 'emotion_distribution' in results['emotions']:
        ax2 = fig.add_subplot(gs[0, 1])
        emotions = list(results['emotions']['emotion_distribution'].keys())
        percentages = list(results['emotions']['emotion_distribution'].values())
        
        bars = ax2.bar(emotions, percentages)
        ax2.set_xlabel('Emotion')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Emotion Distribution')
        ax2.set_ylim(0, max(percentages) * 1.2 if percentages else 100)
        
        # Color bars by emotion
        emotion_colors = {
            'happy': '#FFD700',
            'sad': '#4169E1', 
            'angry': '#DC143C',
            'fear': '#8B008B',
            'disgust': '#228B22',
            'surprise': '#FF69B4',
            'neutral': '#808080'
        }
        for bar, emotion in zip(bars, emotions):
            bar.set_color(emotion_colors.get(emotion, '#808080'))
    
    # 3. Timeline density (segments per minute)
    if 'timeline' in results and results['timeline']:
        ax3 = fig.add_subplot(gs[1, :])
        
        timeline = results['timeline']
        total_duration = max(seg['end'] for seg in timeline)
        
        # Calculate segment density
        bins = np.arange(0, total_duration + 60, 60)  # 1-minute bins
        segment_starts = [seg['start'] for seg in timeline]
        counts, _ = np.histogram(segment_starts, bins=bins)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2 / 60  # Convert to minutes
        ax3.bar(bin_centers, counts, width=0.9, alpha=0.7)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Number of Segments')
        ax3.set_title('Segment Density Over Time')
        ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Summary statistics plot saved to {save_path}")
    
    return fig


def create_visualization_report(
    results: Dict[str, Any],
    audio_data: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
    output_dir: Union[str, Path] = None,
    include_waveform: bool = True,
    include_spectrogram: bool = True
) -> Dict[str, Path]:
    """Create a complete visualization report.
    
    Args:
        results: Analysis results
        audio_data: Optional audio data for waveform/spectrogram
        sample_rate: Sample rate
        output_dir: Output directory for plots
        include_waveform: Include waveform plot
        include_spectrogram: Include spectrogram plot
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    try:
        # 1. Waveform
        if include_waveform and audio_data is not None and sample_rate:
            save_path = output_dir / "waveform.png" if output_dir else None
            fig = plot_waveform(audio_data, sample_rate, save_path=save_path)
            if save_path:
                saved_plots['waveform'] = save_path
            plt.close(fig)
        
        # 2. Spectrogram
        if include_spectrogram and audio_data is not None and sample_rate:
            save_path = output_dir / "spectrogram.png" if output_dir else None
            fig = plot_spectrogram(audio_data, sample_rate, save_path=save_path)
            if save_path:
                saved_plots['spectrogram'] = save_path
            plt.close(fig)
        
        # 3. Speaker timeline
        if 'speakers' in results and results.get('speakers'):
            segments = [s for s in results.get('timeline', []) 
                       if s.get('type') == 'speaker_segment']
            if segments:
                save_path = output_dir / "speaker_timeline.png" if output_dir else None
                fig = plot_speaker_timeline(segments, save_path=save_path)
                if save_path:
                    saved_plots['speaker_timeline'] = save_path
                plt.close(fig)
        
        # 4. Emotion timeline
        if 'emotions' in results and results.get('emotions'):
            predictions = [s for s in results.get('timeline', [])
                         if s.get('type') == 'emotion_segment']
            if predictions:
                save_path = output_dir / "emotion_timeline.png" if output_dir else None
                fig = plot_emotion_timeline(predictions, save_path=save_path)
                if save_path:
                    saved_plots['emotion_timeline'] = save_path
                plt.close(fig)
        
        # 5. Acoustic features
        if 'acoustics' in results and results['acoustics'].get('features'):
            save_path = output_dir / "acoustic_features.png" if output_dir else None
            fig = plot_acoustic_features(
                results['acoustics']['features'],
                save_path=save_path
            )
            if fig and save_path:
                saved_plots['acoustic_features'] = save_path
            if fig:
                plt.close(fig)
        
        # 6. Summary statistics
        save_path = output_dir / "summary.png" if output_dir else None
        fig = plot_summary_statistics(results, save_path=save_path)
        if save_path:
            saved_plots['summary'] = save_path
        plt.close(fig)
        
        logger.info(f"Created {len(saved_plots)} visualization plots")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    return saved_plots