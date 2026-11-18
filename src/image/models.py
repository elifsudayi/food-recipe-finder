"""Image models."""
import torch.nn as nn


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Small CNN trained from scratch: 4 x (conv-BN-ReLU-maxpool) -> GAP -> dropout -> linear."""

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
