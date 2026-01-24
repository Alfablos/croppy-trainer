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


def predict(
    img_path: str,
    architecture: Architecture,
    h: int,
    w: int,
    model: CroppyNet,
    device: Device = Device.CUDA,
    base_weights=visionmodels.ResNet18_Weights.DEFAULT,
) -> torch.Tensor:
    resized: NDArray = architecture.resize_image(
        img_path, h=h, w=w, color=True, resize=True, interpolation=cv2.INTER_AREA
    )
    t = base_weights.transforms()
    pipeline = transformsV2.Compose(
        [
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.Normalize(mean=t.mean, std=t.std),
        ]
    )

    inf_input: torch.Tensor = pipeline(resized)
    input_as_batch = inf_input.unsqueeze(0).to(device.value)

    with torch.inference_mode():
        return model(input_as_batch)
