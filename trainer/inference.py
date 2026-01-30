from data import get_transforms
from common import Device
from torch.utils.hipify.cuda_to_hip_mappings import PYTORCH_SPECIFIC_MAPPINGS
import torch
import torchvision.models as visionmodels
from torchvision.transforms import v2 as transformsV2
import numpy as np
from train import CroppyNet
import cv2
from architecture import Architecture
from numpy.typing import NDArray


@torch.no_grad()
def predict(
    image: NDArray,
    model: CroppyNet,
    device: Device,
    base_weights=visionmodels.ResNet18_Weights.DEFAULT,
) -> torch.Tensor:
    model.eval()

    t = model.weights.transforms()
    transforms = transformsV2.Compose(
        [
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.Normalize(mean=t.mean, std=t.std),
        ]
    )

    img_tensor, _ = transformsV2.ToImage()(image)
    inf_input: torch.Tensor = transforms(img_tensor)

    input_as_batch = inf_input.unsqueeze(0).to(
        device.value
    )  # add a dimension, the model expects a batch
    print(input_as_batch.shape)

    return model(input_as_batch)


def get_image_points(
    image_shape: tuple[int, int, int], coords: torch.Tensor | NDArray
) -> NDArray:
    if isinstance(coords, torch.Tensor):
        coords: NDArray = coords.cpu().numpy()

    coords = coords.squeeze()

    h, w, _c = image_shape
    image_points = []

    # x IS WIDTH!!
    # image_points = [[coords[c] * w, coords[c+1] * h] for c in range(0, 8, 2)]
    for c in range(0, 8, 2):  # x and y at once
        x = coords[c] * w
        y = coords[c + 1] * h
        image_points.append([x, y])

    image_points = np.array(image_points, dtype=np.int32)

    return image_points


def draw_box(corners_coords: NDArray, image: NDArray):
    # Draws the corners
    for xy in corners_coords:
        cv2.circle(center=xy, img=image, color=(255, 0, 0), radius=5, thickness=-1)
    cv2.polylines(image, [corners_coords], isClosed=True, color=(255, 0, 0))
