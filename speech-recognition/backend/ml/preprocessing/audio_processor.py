"""
ml/preprocessing/audio_processor.py — Core audio loading and preprocessing pipeline.

Handles:
1. Loading from bytes/path
2. Resampling
3. Optional stationary noise reduction
4. Silence trimming
5. Volume normalization
6. Padding / Truncating to a fixed length
"""

import io
from pathlib import Path
from typing import Union

import librosa
import numpy as np

from ml.preprocessing.noise_reduction import apply_noise_reduction
from training.config import AudioConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """End-to-end audio preprocessor for SER."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.target_length_samples = int(self.config.sample_rate * self.config.max_duration)

    def process(
        self, 
        audio_source: Union[str, Path, bytes, io.BytesIO], 
        denoise: bool = True
    ) -> tuple[np.ndarray, int]:
        """
        Run the full preprocessing pipeline.
        
        Args:
            audio_source: File path or raw bytes.
            denoise: Whether to apply stationary noise reduction.
            
        Returns:
            (processed_audio_array, sample_rate)
        """
        # 1. Load & Resample
        audio, sr = self._load_audio(audio_source)

        # 2. Denoise
        if denoise:
            audio = apply_noise_reduction(
                audio, 
                sr, 
                prop_decrease=0.8, 
                n_std_thresh_stationary=1.5,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )

        # 3. Trim Silence
        audio = self._trim_silence(audio)

        # 4. Normalize Volume
        audio = self._normalize(audio)

        # 5. Pad or Truncate to fixed length
        audio = self._pad_truncate(audio)

        return audio, sr

    def _load_audio(self, source: Union[str, Path, bytes, io.BytesIO]) -> tuple[np.ndarray, int]:
        """Load audio with librosa, resampling to config target."""
        try:
            if isinstance(source, (str, Path)):
                y, sr = librosa.load(source, sr=self.config.sample_rate, mono=True)
            elif isinstance(source, bytes):
                y, sr = librosa.load(io.BytesIO(source), sr=self.config.sample_rate, mono=True)
            elif isinstance(source, io.BytesIO):
                y, sr = librosa.load(source, sr=self.config.sample_rate, mono=True)
            else:
                raise TypeError(f"Unsupported audio source type: {type(source)}")
                
            # If completely silent or empty, generate a silent array to prevent crashes down the line
            if len(y) == 0 or np.max(np.abs(y)) == 0.0:
                logger.warning("Loaded audio is completely silent or empty.")
                return np.zeros(self.target_length_samples, dtype=np.float32), self.config.sample_rate

            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            # Fallback to silence
            return np.zeros(self.target_length_samples, dtype=np.float32), self.config.sample_rate

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=self.config.top_db,
            frame_length=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        # If trimming removed everything, return original
        if len(audio_trimmed) < self.config.sample_rate * 0.1:  # less than 100ms left
            return audio
        return audio_trimmed

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalize audio to range [-1.0, 1.0]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0.0:
            return audio / max_val
        return audio

    def _pad_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is exactly config.max_duration seconds long."""
        length = len(audio)
        if length > self.target_length_samples:
            # Truncate (keep the center which usually has the most emotion)
            start = (length - self.target_length_samples) // 2
            return audio[start:start + self.target_length_samples]
        elif length < self.target_length_samples:
            # Pad with zeros symmetrically
            pad_total = self.target_length_samples - length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(audio, (pad_left, pad_right), mode='constant')
        return audio
