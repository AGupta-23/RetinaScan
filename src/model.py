"""
model.py — RetinaScan model architecture

Defines build_model(), which returns a ResNet50 configured for binary
DR classification (DR / No DR) via partial fine-tuning:
    - conv1, bn1, layer1, layer2, layer3, avgpool -> frozen
    - layer4                                       -> unfrozen
    - fc replaced with Linear(2048 -> 1)            -> trainable (new layer)

Mirrors dataset.py's pattern (RetinaDataset, preprocess_transform):
import this into your training notebook with:

    from model import build_model
    model = build_model(device)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(device=None, unfreeze_from="layer4", num_outputs=1):
    """
    Build a ResNet50 backbone for binary DR classification.

    Args:
        device (torch.device, optional): Device to move the model to.
            If None, auto-selects cuda if available, else cpu.
        unfreeze_from (str): Name of the block to unfreeze for fine-tuning.
            Default "layer4" (the standard partial fine-tuning choice —
            see Phase 6 README for why: layer4 holds the majority of
            ResNet50's parameters despite being the last stage).
        num_outputs (int): Number of output units on the final layer.
            1 -> single logit + BCEWithLogitsLoss (recommended, default).
            2 -> two logits + CrossEntropyLoss (alternative, softmax style).

    Returns:
        torch.nn.Module: ResNet50 model, moved to `device`, ready to train.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load ImageNet-pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 2. Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze the target block (default: layer4) for fine-tuning
    target_block = getattr(model, unfreeze_from)
    for param in target_block.parameters():
        param.requires_grad = True

    # 4. Replace the final classifier layer for binary output
    #    (new layer is trainable by default — no need to set requires_grad)
    num_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Linear(num_features, num_outputs)

    # 5. Move to device
    model = model.to(device)

    return model


def count_trainable_params(model):
    """
    Returns (trainable_params, total_params, trainable_percent).
    Use this as a sanity check right after build_model() — expect
    ~60-65% trainable when unfreezing layer4 + fc (verified reference:
    14,966,785 / 23,510,081 = 63.7%).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percent = 100 * trainable / total
    return trainable, total, percent


def print_trainable_summary(model):
    """
    Prints a block-by-block trainable=True/False summary, matching the
    Phase 6 notebook output. Useful for quick verification or for
    screenshotting into your report/README.
    """
    for name, child in model.named_children():
        trainable = any(p.requires_grad for p in child.parameters())
        print(f"{name:12s} trainable={trainable}")


if __name__ == "__main__":
    # Quick standalone sanity check: run `python model.py`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    trainable, total, percent = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({percent:.1f}%)")
    print()
    print_trainable_summary(model)