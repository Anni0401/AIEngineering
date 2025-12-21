import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    """

    def __init__(self):
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Classifier
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]

        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 7, 7]

        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # Logits

        return x
