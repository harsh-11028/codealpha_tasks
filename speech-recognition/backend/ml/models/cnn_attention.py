"""
ml/models/cnn_attention.py — CNN with Self-Attention.

Architecture:
- 1D Convolutions for local feature extraction
- Multi-Head Self Attention mechanism to focus on emotionally salient parts
- Fully Connected classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionModel(nn.Module):
    def __init__(self, num_classes: int = 8, input_size: int = 40):
        super().__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.drop_cnn = nn.Dropout(0.3)
        
        # Multi-Head Attention
        # embed_dim must match the output channels of the CNN (256)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.3, batch_first=True)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc_drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, Features, Time)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop_cnn(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        
        # Permute for Attention: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)
        
        # Self-Attention
        # attn_output shape: (Batch, Time, Channels)
        attn_output, _ = self.attention(x, x, x)
        
        # Global Average Pooling over Time dimension
        x = torch.mean(attn_output, dim=1)
        
        x = self.fc_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
