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
    img_path: str,
    model: CroppyNet,
    device: Device,
    base_weights=visionmodels.ResNet18_Weights.DEFAULT,
) -> torch.Tensor:
    resized: NDArray = Architecture.resize_image(
        img_path, h=model.images_height, w=model.images_width, color=True, resize=True, interpolation=cv2.INTER_AREA
    )
    transforms = get_transforms(
        weights=model.weights,
        precision=model.precision,
        train=False
    )

    inf_input: torch.Tensor = transforms(resized)
    input_as_batch = inf_input.unsqueeze(0).to(device.value) # add a dimension, the model expects a batch

    return model(input_as_batch)
