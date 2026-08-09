"""
ml/preprocessing/augmentation.py — Audio data augmentation pipeline.

Provides random transformations (noise, pitch shift, time stretch, gain, crop)
to make the SER models robust to varied acoustic environments.
"""

import random
import numpy as np
import librosa

from training.config import AugmentationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioAugmenter:
    """Applies random transformations to raw audio arrays for data augmentation."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a random set of augmentations.
        Each augmentation has a probability of being applied.
        """
        if not self.config.enabled:
            return audio

        # 1. White Noise Injection (30% chance)
        if random.random() < 0.3:
            audio = self._add_noise(audio)

        # 2. Pitch Shifting (30% chance)
        if random.random() < 0.3:
            audio = self._pitch_shift(audio, sr)

        # 3. Time Stretching (30% chance)
        if random.random() < 0.3:
            audio = self._time_stretch(audio)

        # 4. Random Gain (50% chance)
        if random.random() < 0.5:
            audio = self._random_gain(audio)

        # 5. Random Cropping / Shift (20% chance)
        if random.random() < 0.2:
            audio = self._random_crop(audio)

        return audio

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Inject white noise."""
        noise_amp = self.config.noise_factor * np.random.uniform(0.5, 1.5) * np.amax(np.abs(audio))
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Shift pitch up or down slightly."""
        n_steps = random.choice(self.config.pitch_shift_steps)
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Speed up or slow down without changing pitch."""
        rate = random.choice(self.config.time_stretch_rates)
        return librosa.effects.time_stretch(y=audio, rate=rate)

    def _random_gain(self, audio: np.ndarray) -> np.ndarray:
        """Multiply amplitude by a random factor."""
        min_gain, max_gain = self.config.gain_range
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain

    def _random_crop(self, audio: np.ndarray) -> np.ndarray:
        """Randomly crop a small fraction from the start or end (essentially shifting)."""
        length = len(audio)
        crop_size = int(length * np.random.uniform(0.01, self.config.crop_fraction))
        
        if crop_size == 0:
            return audio
            
        if random.random() < 0.5:
            # Crop start, pad end
            cropped = audio[crop_size:]
            return np.pad(cropped, (0, crop_size), mode='constant')
        else:
            # Crop end, pad start
            cropped = audio[:-crop_size]
            return np.pad(cropped, (crop_size, 0), mode='constant')
