"""
ml/prediction_engine.py — Central inference engine for emotion prediction.

Responsibilities:
  - Load the best trained model from disk
  - Run the full preprocessing + feature extraction + inference pipeline
  - Return structured prediction results with visualization data
  - Provide model metadata

This file is the integration layer between the API and the ML models.
It is populated fully in Phase 6; Phase 2 installs the interface so
the API can import it without crashing.
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from training.config import Config, DEFAULT_CONFIG, EMOTION_LABELS
from utils.logger import get_logger
from utils.helpers import softmax

logger = get_logger(__name__)


class PredictionEngine:
    """
    Singleton engine that loads a trained PyTorch model and exposes
    a simple `predict(audio_bytes) -> dict` interface.

    Thread-safe for concurrent API requests (inference only, no state mutation).
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._model = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and config.training.device == "cuda"
            else "cpu"
        )
        self._model_name: str = "none"
        self._model_version: str = "0.0.0"
        self._is_loaded: bool = False

    @property
    def active_model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    async def load(self) -> None:
        """
        Load the best available model from saved_models/.
        Called once at application startup.

        Priority: best_model.pt → any .pt file in saved_models/
        """
        model_dir = Path(self.config.training.checkpoint_dir)
        best_path = model_dir / "best_model.pt"

        if best_path.exists():
            await self._load_model(best_path)
        else:
            # Try loading any available model
            pts = list(model_dir.glob("*.pt"))
            if pts:
                await self._load_model(pts[0])
            else:
                logger.warning(
                    f"No trained model found in '{model_dir}'. "
                    "Run training/train.py to train a model. "
                    "The engine will return placeholder predictions until then."
                )

    async def _load_model(self, model_path: Path) -> None:
        """Load a PyTorch model checkpoint from disk."""
        try:
            logger.info(f"Loading model from: {model_path}")
            checkpoint = torch.load(
                model_path,
                map_location=self._device,
                weights_only=False,
            )

            # Checkpoint format: {model_state, config, metrics, name, version}
            self._model_name = checkpoint.get("model_name", model_path.stem)
            self._model_version = checkpoint.get("version", "1.0.0")
            architecture = checkpoint.get("architecture", "unknown")

            # Instantiate the correct architecture
            self._model = self._build_model(architecture, checkpoint)
            if self._model:
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._model.to(self._device)
                self._model.eval()
                self._is_loaded = True
                logger.info(
                    f"✅ Model loaded: {self._model_name} v{self._model_version} "
                    f"[{architecture}] on {self._device}"
                )
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _build_model(self, architecture: str, checkpoint: dict) -> Optional[torch.nn.Module]:
        """Instantiate the correct model class based on architecture name."""
        try:
            model_config = checkpoint.get("model_config", {})
            num_classes = checkpoint.get("num_classes", self.config.num_classes)
            input_size = checkpoint.get("input_size", 40)

            if architecture == "cnn":
                from ml.models.cnn_model import CNNModel
                return CNNModel(num_classes=num_classes, input_size=input_size)

            elif architecture == "cnn_lstm":
                from ml.models.cnn_lstm_model import CNNLSTMModel
                return CNNLSTMModel(num_classes=num_classes, input_size=input_size)

            elif architecture == "bilstm":
                from ml.models.bilstm_model import BiLSTMModel
                return BiLSTMModel(num_classes=num_classes, input_size=input_size)

            elif architecture == "cnn_attention":
                from ml.models.cnn_attention import CNNAttentionModel
                return CNNAttentionModel(num_classes=num_classes, input_size=input_size)

            elif architecture == "wav2vec2":
                from ml.models.wav2vec_model import Wav2Vec2EmotionModel
                return Wav2Vec2EmotionModel(num_classes=num_classes)

            else:
                logger.error(f"Unknown model architecture: '{architecture}'")
                return None

        except ImportError as e:
            logger.warning(f"Could not import model class for '{architecture}': {e}")
            return None

    def predict(self, audio_bytes: bytes) -> dict:
        """
        Full prediction pipeline.

        Args:
            audio_bytes: Raw audio file bytes (WAV, MP3, etc.)

        Returns:
            Structured dictionary with emotion, probabilities, and visualization data.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "No model loaded. Please train a model first using training/train.py."
            )

        import io
        start = time.perf_counter()

        # ── Step 1: Preprocess audio ──────────────────────────────────────
        from ml.preprocessing.audio_processor import AudioProcessor
        processor = AudioProcessor(self.config.audio)
        audio_array, sr = processor.process(io.BytesIO(audio_bytes))
        
        duration_seconds = len(audio_array) / sr
        is_wav2vec2 = (self._model_name == "wav2vec2")

        # ── Step 2 & 3: Extract features and Prepare tensor ───────────────
        if is_wav2vec2:
            # Wav2Vec2 takes raw audio directly
            feature_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(self._device)
            input_tensor_for_xai = feature_tensor.clone().requires_grad_(True)
        else:
            # Other models take 2D extracted features
            from ml.feature_extraction.extractor import FeatureExtractor
            extractor = FeatureExtractor(self.config)
            features = extractor.extract(audio_array, sr)  # shape: (n_features, T)
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)
            input_tensor_for_xai = feature_tensor.clone().requires_grad_(True)

        # ── Step 4: Inference ──────────────────────────────────────────────
        inference_start = time.perf_counter()
        
        # We run inference with gradients enabled for XAI saliency map
        self._model.eval()
        logits = self._model(input_tensor_for_xai)  # (1, num_classes)
        
        inference_ms = (time.perf_counter() - inference_start) * 1000

        # ── Step 5: Post-process ──────────────────────────────────────────
        logits_np = logits.squeeze(0).detach().cpu().numpy()
        probs = softmax(logits_np)

        predicted_idx = int(np.argmax(probs))
        predicted_emotion = EMOTION_LABELS[predicted_idx]
        confidence = float(probs[predicted_idx])

        probabilities = {
            EMOTION_LABELS[i]: float(probs[i])
            for i in range(len(EMOTION_LABELS))
        }

        # ── Step 6: Explainable AI (XAI) Feature Importance ───────────────
        # Simple gradient-based saliency: How much does each input feature affect the predicted class?
        logits[0, predicted_idx].backward()
        saliency = input_tensor_for_xai.grad.data.abs().squeeze(0).cpu().numpy()
        
        feature_importance = self._compute_feature_importance(saliency, is_wav2vec2)

        # ── Step 7: Build visualization data ──────────────────────────────
        # Downsample waveform for frontend display (max 512 points)
        waveform_data = _downsample(audio_array, 512).tolist()

        # MFCC heatmap data (first 40 coefficients, downsampled in time)
        from ml.feature_extraction.extractor import compute_mfcc_visual
        mfcc_visual = compute_mfcc_visual(audio_array, sr, self.config.audio)
        spectrogram_visual = compute_spectrogram_visual(audio_array, sr, self.config.audio)

        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "waveform_data": waveform_data,
            "mfcc_data": mfcc_visual,
            "spectrogram_data": spectrogram_visual,
            "feature_importance": feature_importance,
            "duration_seconds": duration_seconds,
            "sample_rate": sr,
            "inference_time_ms": inference_ms,
            "model_name": self._model_name,
            "model_version": self._model_version,
        }

    def _compute_feature_importance(self, saliency: np.ndarray, is_wav2vec2: bool) -> dict:
        """
        Aggregate gradient saliency into meaningful feature groups.
        """
        if is_wav2vec2:
            # Saliency is over the 1D time domain. We can't easily map to MFCC.
            return {
                "Time Frame 0-25%": float(np.mean(saliency[:len(saliency)//4])),
                "Time Frame 25-50%": float(np.mean(saliency[len(saliency)//4:len(saliency)//2])),
                "Time Frame 50-75%": float(np.mean(saliency[len(saliency)//2:3*len(saliency)//4])),
                "Time Frame 75-100%": float(np.mean(saliency[3*len(saliency)//4:])),
            }
        else:
            # Saliency is (n_features, T)
            # Roughly map row indices back to feature types (simplification)
            # In our extractor: Mel(128) -> MFCC(40) -> Chroma(12) -> etc.
            # For this MVP, we just take the mean saliency of chunks of rows
            total_rows = saliency.shape[0]
            if total_rows >= 111: # If standard config is used
                return {
                    "Mel Spectrogram": float(np.mean(saliency[:128])),
                    "MFCC": float(np.mean(saliency[128:168])),
                    "Chroma": float(np.mean(saliency[168:180])),
                    "Energy / ZCR": float(np.mean(saliency[180:])),
                }
            else:
                return {
                    "Low Frequency Features": float(np.mean(saliency[:total_rows//3])),
                    "Mid Frequency Features": float(np.mean(saliency[total_rows//3:2*total_rows//3])),
                    "High Frequency Features": float(np.mean(saliency[2*total_rows//3:])),
                }


# ── Visualization helpers ─────────────────────────────────────────────────────
def _downsample(audio: np.ndarray, target_points: int) -> np.ndarray:
    """Downsample audio array to a fixed number of points for visualization."""
    if len(audio) <= target_points:
        return audio
    indices = np.linspace(0, len(audio) - 1, target_points, dtype=int)
    return audio[indices]


def compute_spectrogram_visual(
    audio: np.ndarray, sr: int, audio_config
) -> list[list[float]]:
    """
    Compute a log-mel spectrogram for frontend visualization.
    Returns a 2D list: [n_mels][time_frames]
    """
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=audio_config.n_mels,
            hop_length=audio_config.hop_length,
            n_fft=audio_config.n_fft,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.tolist()
    except Exception as e:
        logger.warning(f"Spectrogram visual failed: {e}")
        return []
