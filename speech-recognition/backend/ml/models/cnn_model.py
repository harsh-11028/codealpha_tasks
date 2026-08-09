"""
ml/models/cnn_model.py — 2D Convolutional Neural Network for Audio Features.

Architecture:
- Input shape: (Batch, Channels=1, Features, TimeFrames)
- 3 Conv2D blocks with BatchNorm, ReLU, and MaxPooling
- Global Average Pooling
- Fully Connected classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes: int = 8, input_size: int = 40):
        """
        Args:
            num_classes: Number of emotion classes
            input_size: Number of features (e.g., n_mfcc, or total concatenated features)
                        This defines the height of the input 2D array.
        """
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(0.2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(0.3)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.drop3 = nn.Dropout2d(0.4)

        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape expected: (Batch, Features, Time)
        # Add channel dimension: (Batch, 1, Features, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.fc_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
