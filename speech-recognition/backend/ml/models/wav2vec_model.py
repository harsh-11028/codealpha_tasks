"""
ml/models/wav2vec_model.py — Transfer Learning with Wav2Vec 2.0.

Architecture:
- HuggingFace Wav2Vec2Base Model (feature extractor)
- Takes raw audio waveform (1D array) as input, rather than extracted 2D features
- Fully Connected classifier on top of the pooled transformer output
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from utils.logger import get_logger

logger = get_logger(__name__)

class Wav2Vec2EmotionModel(nn.Module):
    def __init__(self, num_classes: int = 8, model_name: str = "facebook/wav2vec2-base", freeze_layers: int = 6):
        """
        Args:
            num_classes: Number of emotion classes
            model_name: HuggingFace model hub name
            freeze_layers: Number of transformer layers to freeze to prevent catastrophic forgetting
        """
        super().__init__()
        
        logger.info(f"Loading Wav2Vec2 model: {model_name}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze the CNN feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Freeze specified number of transformer layers
        if freeze_layers > 0:
            for i in range(freeze_layers):
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False
            logger.info(f"Froze CNN feature extractor and first {freeze_layers} transformer layers.")
            
        # Wav2Vec2 base hidden size is 768
        hidden_size = self.wav2vec2.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape expected: (Batch, Audio_Length)
        # Note: Wav2Vec2 takes raw audio, not 2D features like the other models!
        
        # Squeeze out the feature dimension if it was accidentally passed as (Batch, 1, Time)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        outputs = self.wav2vec2(x)
        
        # We take the mean across the time dimension of the last hidden state
        # hidden_states shape: (Batch, SequenceLength, HiddenSize)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
        logits = self.classifier(pooled_output)
        
        return logits
