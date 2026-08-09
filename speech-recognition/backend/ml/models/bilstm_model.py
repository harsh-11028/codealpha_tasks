"""
ml/models/bilstm_model.py — Bidirectional LSTM Neural Network.

Architecture:
- Pure recurrent architecture for sequence modeling
- Bidirectional LSTM layers
- Fully Connected classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, num_classes: int = 8, input_size: int = 40):
        super().__init__()
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=256, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier
        self.fc1 = nn.Linear(256 * 2, 128)
        self.fc_drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape expected: (Batch, Features, Time)
        # LSTM needs (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        
        _, (h_n, _) = self.lstm(x)
        
        # Concat last hidden states of forward and backward directions from the top layer
        h_n_concat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        x = self.fc_drop(F.relu(self.fc1(h_n_concat)))
        x = self.fc2(x)
        
        return x
