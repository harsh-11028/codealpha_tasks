"""
Model selector — benchmarks all available trained models and picks the best.

Strategies:
  - Character-level task: selects by top-1 accuracy on validation set
  - Word-level task: selects by Character Error Rate (lower is better)
  - Auto mode: runs quick benchmark on a sample batch and picks the winner

Usage:
    from models.utils.model_selector import ModelSelector
    selector = ModelSelector(cfg)
    best = selector.select_best(task='character')
    result = selector.predict(image, task='character')
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.training.config import Config, config as default_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "cnn_basic":     "models.architectures.cnn_basic.CNNBasic",
    "cnn_batchnorm": "models.architectures.cnn_batchnorm.CNNBatchNorm",
    "residual_cnn":  "models.architectures.residual_cnn.ResidualCNN",
    "crnn":          "models.architectures.crnn.CRNN",
    "vit":           "models.architectures.vit.VisionTransformer",
}

WORD_MODELS = {"crnn"}      # models suitable for word/sentence recognition
CHAR_MODELS = {"cnn_basic", "cnn_batchnorm", "residual_cnn", "vit"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_class(dotted_path: str):
    """Dynamically import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolve the compute device.

    Args:
        device_str: 'auto' | 'cpu' | 'cuda' | 'mps'

    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def get_checkpoint_path(model_name: str, cfg: Config) -> Path:
    """Return the expected .pt checkpoint path for a model."""
    return Path(cfg.training.checkpoint_dir) / f"best_{model_name}.pt"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    cfg: Config,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Instantiate a model by name using config parameters.

    Args:
        model_name:  One of the MODEL_REGISTRY keys.
        cfg:         Project configuration.
        num_classes: Override class count.

    Returns:
        Unloaded (randomly initialized) PyTorch model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = _import_class(MODEL_REGISTRY[model_name])
    nc = num_classes or cfg.training.num_classes

    kwargs: dict = {"num_classes": nc}

    if model_name in ("cnn_basic", "cnn_batchnorm", "residual_cnn"):
        kwargs.update({
            "in_channels": cfg.cnn.in_channels,
            "base_filters": cfg.cnn.base_filters,
        })
        if model_name == "cnn_basic":
            kwargs["dropout_rate"] = cfg.cnn.dropout_rate
            kwargs["fc_hidden_size"] = cfg.cnn.fc_hidden_size

    elif model_name == "crnn":
        kwargs.update({
            "in_channels":  cfg.crnn.in_channels,
            "cnn_out_ch":   cfg.crnn.cnn_out_channels,
            "rnn_hidden":   cfg.crnn.rnn_hidden_size,
            "rnn_layers":   cfg.crnn.rnn_num_layers,
            "rnn_dropout":  cfg.crnn.rnn_dropout,
        })

    elif model_name == "vit":
        h, w = cfg.preprocessing.image_size
        kwargs.update({
            "image_size":        h,
            "patch_size":        cfg.vit.patch_size,
            "in_channels":       cfg.vit.in_channels,
            "embed_dim":         cfg.vit.embed_dim,
            "num_heads":         cfg.vit.num_heads,
            "num_layers":        cfg.vit.num_layers,
            "mlp_ratio":         cfg.vit.mlp_ratio,
            "dropout_rate":      cfg.vit.dropout_rate,
            "attention_dropout": cfg.vit.attention_dropout,
        })

    model = cls(**kwargs)
    logger.debug(
        "Built model '%s' with %d parameters.",
        model_name, sum(p.numel() for p in model.parameters()),
    )
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> Dict:
    """
    Load model weights from a checkpoint file.

    Args:
        model:           Model instance to load weights into.
        checkpoint_path: Path to .pt checkpoint.
        device:          Target device.
        strict:          Strict state dict matching.

    Returns:
        Checkpoint metadata dict (epoch, metrics, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return checkpoint


# ---------------------------------------------------------------------------
# Main model selector
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Manages multiple trained models and selects the best for each task.

    Usage:
        selector = ModelSelector(cfg)
        selector.load_all()
        best_name, best_model = selector.select_best(task='character')
        logits = selector.predict_single(image_tensor, model_name='vit')
    """

    def __init__(self, cfg: Config = default_config) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self._models: Dict[str, nn.Module] = {}
        self._metrics: Dict[str, Dict[str, float]] = {}

    def load_model(
        self,
        model_name: str,
        strict: bool = True,
    ) -> Optional[nn.Module]:
        """
        Load a single model from its checkpoint.

        Returns None if checkpoint does not exist.

        Args:
            model_name: Registry key for the model.
            strict:     Strict state dict loading.

        Returns:
            Loaded model on self.device, or None.
        """
        ckpt_path = get_checkpoint_path(model_name, self.cfg)
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s", ckpt_path)
            return None

        model = build_model(model_name, self.cfg)
        checkpoint = load_checkpoint(model, ckpt_path, self.device, strict)
        self._models[model_name] = model
        self._metrics[model_name] = checkpoint.get("metrics", {})
        return model

    def load_all(self) -> None:
        """Attempt to load all registered models."""
        for name in MODEL_REGISTRY:
            self.load_model(name, strict=False)
        logger.info(
            "ModelSelector: loaded %d / %d models.",
            len(self._models), len(MODEL_REGISTRY),
        )

    def get_loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def select_best(
        self,
        task: str = "character",
        val_loader=None,
    ) -> Tuple[str, nn.Module]:
        """
        Select the best model for the given task.

        If metrics are already stored in checkpoints, uses those directly.
        Otherwise runs a quick benchmark on val_loader.

        Args:
            task:       'character' | 'word' | 'sentence'
            val_loader: DataLoader for benchmarking (optional).

        Returns:
            (best_model_name, best_model)

        Raises:
            RuntimeError if no models are loaded.
        """
        candidates = (
            WORD_MODELS if task in ("word", "sentence")
            else CHAR_MODELS
        )
        available = [n for n in self._models if n in candidates]

        if not available:
            # Fall back to any loaded model
            available = list(self._models.keys())

        if not available:
            raise RuntimeError(
                "No models loaded. Call load_all() or load_model() first."
            )

        # Use stored metrics if available
        scored: Dict[str, float] = {}
        for name in available:
            m = self._metrics.get(name, {})
            if task in ("word", "sentence"):
                # Lower CER is better → negate
                scored[name] = -m.get("cer", float("inf"))
            else:
                scored[name] = m.get("accuracy", 0.0)

        # Benchmark if no useful stored metrics
        if all(v == 0.0 for v in scored.values()) and val_loader is not None:
            logger.info("No stored metrics — running benchmark on validation set.")
            scored = self._benchmark(available, val_loader, task)

        best_name = max(scored, key=lambda k: scored[k])
        logger.info(
            "Best model for task='%s': %s (score=%.4f)",
            task, best_name, scored[best_name],
        )
        return best_name, self._models[best_name]

    def _benchmark(
        self,
        model_names: List[str],
        val_loader,
        task: str,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """Quick benchmark of models on a subset of the validation set."""
        from models.utils.metrics import accuracy

        scores: Dict[str, float] = {}
        for name in model_names:
            model = self._models[name]
            model.eval()
            correct = total = 0
            t_start = time.perf_counter()

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= max_batches:
                        break
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    try:
                        logits = model(images)
                    except Exception as e:
                        logger.warning("Model %s failed during benchmark: %s", name, e)
                        break
                    if isinstance(logits, torch.Tensor) and logits.ndim == 2:
                        top1, = accuracy(logits, labels, topk=(1,))
                        correct += top1 * images.size(0) / 100
                        total += images.size(0)

            elapsed = time.perf_counter() - t_start
            acc = (correct / max(total, 1)) * 100
            logger.info(
                "Benchmark %s: acc=%.2f%% | time=%.2fs", name, acc, elapsed
            )
            scores[name] = acc

        return scores

    def predict_single(
        self,
        image_tensor: torch.Tensor,
        model_name: str = "auto",
        task: str = "character",
    ) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a single preprocessed image tensor.

        Args:
            image_tensor: Float tensor of shape (1, 1, H, W).
            model_name:   Specific model to use, or 'auto' for best.
            task:         'character' | 'word' | 'sentence'

        Returns:
            (logits, confidence): Logit tensor and top-1 confidence score.
        """
        if model_name == "auto":
            if not self._models:
                raise RuntimeError("No models loaded.")
            model_name = list(self._models.keys())[0]

        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name!r} not loaded.")

        model.eval()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = model(image_tensor)

        import torch.nn.functional as F
        probs = F.softmax(logits, dim=-1)
        confidence = float(probs.max().item())
        return logits, confidence

    def get_model_info(self) -> List[Dict]:
        """Return info dict for all loaded models (for API /model-info)."""
        info = []
        for name, model in self._models.items():
            n_params = sum(p.numel() for p in model.parameters())
            info.append({
                "name": name,
                "parameters": n_params,
                "metrics": self._metrics.get(name, {}),
                "device": str(self.device),
            })
        return info
