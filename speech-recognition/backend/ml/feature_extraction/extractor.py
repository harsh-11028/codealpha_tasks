"""
ml/feature_extraction/extractor.py — Acoustic feature extraction.

Extracts a comprehensive set of audio features:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Delta and Delta-Delta MFCCs
- Mel Spectrogram
- Chroma
- Spectral Contrast
- Spectral Centroid
- Tonnetz
- RMS Energy
- Zero Crossing Rate (ZCR)
- Pitch
"""

import librosa
import numpy as np

from training.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extracts and concatenates acoustic features into a single 2D array."""

    def __init__(self, config: Config):
        self.config = config
        self.feat_cfg = config.features
        self.aud_cfg = config.audio

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract configured features.
        
        Args:
            audio: 1D numpy array of audio samples (preprocessed, fixed length)
            sr: Sample rate
            
        Returns:
            2D numpy array of concatenated features shape (N_features, TimeFrames)
        """
        features_list = []

        # Common kwargs for librosa
        stft_kwargs = {
            "n_fft": self.aud_cfg.n_fft,
            "hop_length": self.aud_cfg.hop_length
        }

        # Pre-compute STFT for efficiency since multiple features use it
        stft = np.abs(librosa.stft(audio, **stft_kwargs))

        # 1. Mel Spectrogram
        if self.feat_cfg.use_mel_spectrogram:
            mel = librosa.feature.melspectrogram(
                S=stft**2, 
                sr=sr, 
                n_mels=self.aud_cfg.n_mels,
                fmin=self.aud_cfg.fmin,
                fmax=self.aud_cfg.fmax
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            features_list.append(log_mel)

        # 2. MFCC (Depends on Mel Spectrogram)
        if self.feat_cfg.use_mfcc:
            # Recompute to ensure exact configuration match if mel config differs, 
            # but usually passing S=log_mel works. For safety, recompute via librosa
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.aud_cfg.n_mfcc, 
                hop_length=self.aud_cfg.hop_length,
                n_fft=self.aud_cfg.n_fft
            )
            features_list.append(mfcc)

            # 3. Delta MFCC
            if self.feat_cfg.use_delta_mfcc:
                delta_mfcc = librosa.feature.delta(mfcc)
                features_list.append(delta_mfcc)

            # 4. Delta-Delta MFCC
            if self.feat_cfg.use_delta2_mfcc:
                delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                features_list.append(delta2_mfcc)

        # 5. Chroma
        if self.feat_cfg.use_chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
            features_list.append(chroma)

        # 6. Spectral Contrast
        if self.feat_cfg.use_spectral_contrast:
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
            features_list.append(contrast)

        # 7. Spectral Centroid
        if self.feat_cfg.use_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
            features_list.append(centroid)

        # 8. Tonnetz
        if self.feat_cfg.use_tonnetz:
            # Tonnetz requires harmonic component
            y_harmonic = librosa.effects.harmonic(audio)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            features_list.append(tonnetz)

        # 9. RMS Energy
        if self.feat_cfg.use_rms:
            rms = librosa.feature.rms(S=stft)
            features_list.append(rms)

        # 10. Zero Crossing Rate
        if self.feat_cfg.use_zcr:
            zcr = librosa.feature.zero_crossing_rate(
                audio, 
                frame_length=self.aud_cfg.n_fft, 
                hop_length=self.aud_cfg.hop_length
            )
            features_list.append(zcr)

        # 11. Pitch (Fundamental Frequency)
        if self.feat_cfg.use_pitch:
            f0, _, _ = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), 
                sr=sr,
                frame_length=self.aud_cfg.n_fft,
                hop_length=self.aud_cfg.hop_length
            )
            # Replace NaNs (unvoiced frames) with 0
            f0 = np.nan_to_num(f0)
            features_list.append(f0.reshape(1, -1))

        # Concatenate all features along the feature dimension (axis=0)
        # Result shape: (Total_Features, Time_Frames)
        if not features_list:
            raise ValueError("No features were enabled in configuration.")
            
        combined_features = np.vstack(features_list)
        
        # Normalize combined features (Standardization: mean=0, std=1 per feature)
        # This is critical for neural network convergence
        mean = np.mean(combined_features, axis=1, keepdims=True)
        std = np.std(combined_features, axis=1, keepdims=True)
        combined_features = (combined_features - mean) / (std + 1e-8)

        return combined_features


def compute_mfcc_visual(audio: np.ndarray, sr: int, audio_config) -> list[list[float]]:
    """
    Compute downsampled MFCC for frontend visualization (heatmap).
    Returns a 2D list: [n_mfcc][time_frames]
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=audio_config.n_mfcc, 
            hop_length=audio_config.hop_length,
            n_fft=audio_config.n_fft
        )
        
        # Normalize for visualization [0, 1]
        mfcc_min = mfcc.min()
        mfcc_max = mfcc.max()
        if mfcc_max > mfcc_min:
            mfcc_norm = (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
        else:
            mfcc_norm = mfcc
            
        # Downsample time dimension if too large for frontend (e.g., > 100 frames)
        max_frames = 100
        if mfcc_norm.shape[1] > max_frames:
            indices = np.linspace(0, mfcc_norm.shape[1] - 1, max_frames, dtype=int)
            mfcc_norm = mfcc_norm[:, indices]
            
        return mfcc_norm.tolist()
    except Exception as e:
        logger.warning(f"MFCC visual computation failed: {e}")
        return []
