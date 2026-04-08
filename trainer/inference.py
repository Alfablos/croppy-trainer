from common import Device
import torch
from torchvision.transforms import v2 as transformsV2
import numpy as np
from train import CroppyNet
from utils import get_resize_params
import cv2
from numpy.typing import NDArray


@torch.no_grad()
def predict(
    image: NDArray,
    model: CroppyNet,
    device: Device,
) -> NDArray:
    model.eval()

    t = model.weights.transforms()
    transforms = transformsV2.Compose(
        [
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.Normalize(mean=t.mean, std=t.std),
        ]
    )

    img_tensor = transformsV2.ToImage()(image)
    inf_input: torch.Tensor = transforms(img_tensor)

    input_as_batch = inf_input.unsqueeze(0).to(device.value)

    return model(input_as_batch)


def get_image_points(
    image_shape: tuple[int, int, int],
    target_shape: tuple[int, int],
    coords: torch.Tensor | NDArray,
) -> NDArray:
    """
    Map model output (normalized [0,1] in the padded model-input canvas) back
    to original-image pixel coordinates.

    The training pipeline (``Architecture._transform_resnet``) resizes the
    original image to fit inside ``target_shape`` preserving aspect ratio,
    then zero-pads the shorter side. Labels are scaled by the same factor
    and offset by the padding, so the model learns coordinates in the
    *padded* canvas. Inference must invert that exact transform.

    Args:
        image_shape: ``(orig_h, orig_w, channels)`` of the original image.
        target_shape: ``(target_h, target_w)`` — the model's input dims.
        coords: model output, normalized [0,1] in the padded canvas.
    """
    coords_np: NDArray = (
        coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    )
    coords_np = coords_np.squeeze()

    orig_h, orig_w = image_shape[:2]
    target_h, target_w = target_shape

    # Recompute the same geometry the forward (training) pass used so the
    # inverse is guaranteed consistent with how labels were positioned.
    _, _, pad_h, pad_w, scale = get_resize_params(
        original_h=orig_h, original_w=orig_w, target_h=target_h, target_w=target_w
    )
    pad_left = pad_w // 2  # matches _transform_resnet's offset
    pad_top = pad_h // 2

    image_points = []
    for c in range(0, 8, 2):  # x and y at once
        # 1. normalized [0,1] padded → padded pixel
        padded_x = coords_np[c] * target_w
        padded_y = coords_np[c + 1] * target_h
        # 2. undo pad offset
        unpadded_x = padded_x - pad_left
        unpadded_y = padded_y - pad_top
        # 3. undo resize scale → original-image pixel coordinates
        x = unpadded_x / scale
        y = unpadded_y / scale
        # Clamp to image bounds (small under/overshoot is possible at the edges)
        x = max(0, min(orig_w, x))
        y = max(0, min(orig_h, y))
        image_points.append([x, y])

    return np.array(image_points, dtype=np.int32)


def draw_box(corners_coords: NDArray, image: NDArray):
    # Draws the corners
    for xy in corners_coords:
        print(f"Drawing corner at ({xy[0]}, {xy[1]})")
        cv2.circle(center=xy, img=image, color=(255, 0, 0), radius=5, thickness=5)
    cv2.polylines(image, [corners_coords], isClosed=True, color=(255, 0, 0))
