"""
ml/preprocessing/noise_reduction.py — Advanced audio noise reduction.

Uses the 'noisereduce' library which implements spectral gating to
remove background stationary noise (e.g., microphone hiss, hums).
"""

import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def apply_noise_reduction(
    audio: np.ndarray, 
    sr: int, 
    prop_decrease: float = 1.0, 
    n_std_thresh_stationary: float = 1.5,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Apply stationary noise reduction to the audio signal.
    
    Args:
        audio: 1D numpy array of audio data
        sr: Sample rate
        prop_decrease: Proportion to reduce noise (0.0 to 1.0)
        n_std_thresh_stationary: Threshold for stationary noise detection
        n_fft: FFT window size
        hop_length: FFT hop length
        
    Returns:
        Denoised 1D numpy array
    """
    try:
        import noisereduce as nr
        
        if len(audio) < n_fft:
            # Audio too short for this n_fft, return as is
            return audio

        reduced_audio = nr.reduce_noise(
            y=audio, 
            sr=sr, 
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh_stationary,
            n_fft=n_fft,
            hop_length=hop_length,
            n_jobs=1, # Keep 1 for stable async API serving
        )
        return reduced_audio
        
    except ImportError:
        logger.warning("'noisereduce' not installed. Skipping noise reduction.")
        return audio
    except Exception as e:
        logger.error(f"Noise reduction failed: {e}. Returning original audio.")
        return audio
