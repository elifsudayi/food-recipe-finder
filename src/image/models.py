"""Image models."""
import torch.nn as nn
from torchvision import models


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


# ---------------------------------------------------------------- VGG16 ----
# torchvision VGG16 `features` layer indices: block4 starts at 17, block5 at 24.
BLOCK4_START, BLOCK5_START = 17, 24


def build_vgg16(num_classes=10, dropout=0.5, pretrained=True):
    """ImageNet-pretrained VGG16 convolutional base + a new small classifier head.

    Input must be 224x224 (the head expects 512*7*7 = 25088 features).
    """
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.vgg16(weights=weights)
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(25088, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return m


def set_trainable(model, unfreeze_from=None):
    """Freeze everything, then unfreeze the head and (optionally) conv layers
    from `features[unfreeze_from:]` onward.

    unfreeze_from=None          -> feature extraction (only the new head trains)
    unfreeze_from=BLOCK5_START  -> fine-tune conv block 5 + head
    unfreeze_from=BLOCK4_START  -> fine-tune conv blocks 4-5 + head
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if unfreeze_from is not None:
        for i, layer in enumerate(model.features):
            if i >= unfreeze_from:
                for p in layer.parameters():
                    p.requires_grad = True


def head_and_backbone_params(model):
    """(head params, trainable backbone params) for per-group learning rates."""
    head = list(model.classifier.parameters())
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    return head, backbone
