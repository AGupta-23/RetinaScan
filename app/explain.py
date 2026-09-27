"""Grad-CAM overlay and matched square display for the two image panes."""

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor):
    """
    Undo ImageNet normalisation and return a float32 HxWx3 array in [0, 1]
    suitable for show_cam_on_image().
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HxWx3
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def pad_for_view(image, size=720, inset=0.04):
    """
    Show the full photograph with margin. Does not stretch or crop, so the
    circular fundus is not clipped by the frame.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    canvas = Image.new("RGB", (size, size), (7, 8, 10))
    inner = int(size * (1.0 - 2.0 * inset))
    w, h = image.size
    scale = min(inner / w, inner / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    fitted = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def generate_gradcam(model, input_tensor, device, prob):
    """
    Generate a Grad-CAM heatmap overlay.

    Target layer: model.layer4[-1]  (last conv block of ResNet50)
    Target class: BinaryClassifierOutputTarget  (single-logit output)

    Returns an HxWx3 uint8 RGB overlay image.
    """
    target_layers = [model.layer4[-1]]
    # BinaryClassifierOutputTarget is correct for single-logit output;
    # do NOT use ClassifierOutputTarget (that is for multi-class softmax).
    targets = [BinaryClassifierOutputTarget(None)]

    input_on_device = input_tensor.to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_on_device, targets=targets)

    grayscale_cam = grayscale_cam[0]       # HxW float in [0, 1]
    rgb_image = denormalize(input_tensor)  # HxWx3 float in [0, 1]

    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return overlay                         # HxWx3 uint8
