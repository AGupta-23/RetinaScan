"""Grad-CAM overlay and matched square display for the two image panes."""

import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


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


def generate_gradcam(model, input_tensor, device, prob, original_image):
    """
    Generate a Grad-CAM heatmap overlay, resized back onto the ORIGINAL
    (non-stretched) image dimensions before overlaying.

    Grad-CAM itself is computed on the model's actual input (a 224x224
    tensor produced by preprocess_transform, which uses a plain
    aspect-ratio-breaking resize -- transforms.Resize((224, 224))). If the
    resulting heatmap were overlaid on that same stretched square, it would
    align correctly with the model's own input space but NOT with the
    original, unstretched photo shown in the adjacent "Your photo" panel --
    making correctly-located attention look like it's pointing at the wrong
    anatomical region. Resizing the heatmap onto original_image's true
    dimensions (instead of denormalizing the stretched tensor) fixes that
    display misalignment.

    Target layer: model.layer4[-1]  (last conv block of ResNet50)
    Target class: BinaryClassifierOutputTarget  (single-logit output)

    Args:
        original_image: the original PIL image (pre-preprocessing), used
            only to recover the true aspect ratio/dimensions for display.

    Returns an HxWx3 uint8 RGB overlay image, matching original_image's
    dimensions.
    """
    target_layers = [model.layer4[-1]]
    # BinaryClassifierOutputTarget is correct for single-logit output;
    # do NOT use ClassifierOutputTarget (that is for multi-class softmax).
    targets = [BinaryClassifierOutputTarget(None)]

    input_on_device = input_tensor.to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_on_device, targets=targets)

    grayscale_cam = grayscale_cam[0]  # 224x224 float32, in [0, 1]

    # Recover the original (unstretched) image as a float32 RGB array in
    # [0, 1] -- this is what the heatmap gets overlaid onto, NOT the
    # denormalized stretched tensor.
    original_rgb = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
    orig_h, orig_w = original_rgb.shape[:2]

    # Resize the heatmap from the model's 224x224 space onto the original
    # image's true dimensions so the two panels share the same geometry.
    grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))

    overlay = show_cam_on_image(original_rgb, grayscale_cam_resized, use_rgb=True)
    return overlay  # HxWx3 uint8, same dimensions as original_image