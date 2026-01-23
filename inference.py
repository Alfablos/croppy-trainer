from common import Device
from torch.utils.hipify.cuda_to_hip_mappings import PYTORCH_SPECIFIC_MAPPINGS
import torch
from train import CroppyNet
import cv2
from architecture import Architecture
from numpy.typing import NDArray



def predict(img_path: str, architecture: Architecture, h: int, w: int, model: CroppyNet, device: Device = Device.CUDA) -> torch.Tensor:
    resized: NDArray = architecture.resize_image(
        img_path, h=h, w=w, color=True, resize=True, interpolation=cv2.INTER_AREA
    )
    resized_tensor = torch.from_numpy(resized).to(device=device.value)
    with torch.inference_mode():
        return model(resized)
    
    
    
    
    