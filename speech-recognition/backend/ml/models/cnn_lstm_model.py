"""
ml/models/cnn_lstm_model.py — Hybrid CNN + LSTM Neural Network.

Architecture:
- 1D Convolutions over the time axis (extract local temporal patterns)
- LSTM to capture long-term sequential dependencies
- Fully Connected classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes: int = 8, input_size: int = 40):
        """
        Args:
            num_classes: Number of emotion classes
            input_size: Number of features per time step (height of input array)
        """
        super().__init__()
        
        # We treat the feature dimension as the "channels" for Conv1d
        # Input shape: (Batch, Features, Time)
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)
        
        # LSTM layer
        # PyTorch LSTM expects input shape: (Batch, Time, Features) if batch_first=True
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier (Bidirectional means hidden_size * 2)
        self.fc1 = nn.Linear(128 * 2, 128)
        self.fc_drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, Features, Time)
        
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        
        # Prepare for LSTM: Permute to (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        
        # LSTM returns (output, (h_n, c_n))
        # h_n shape: (num_layers * num_directions, Batch, HiddenSize)
        _, (h_n, _) = self.lstm(x)
        
        # Get the final hidden state of both directions from the top layer
        # h_n[-2, :, :] is forward, h_n[-1, :, :] is backward
        h_n_concat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        x = self.fc_drop(F.relu(self.fc1(h_n_concat)))
        x = self.fc2(x)
        
        return x
