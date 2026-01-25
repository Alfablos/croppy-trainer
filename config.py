import torch
from torchvision.transforms import v2 as transformsV2



train_cpu_transforms = transformsV2.Compose([
    # photometric
    transformsV2.ToImage(),
    transformsV2.GaussianBlur(kernel_size=(15, 31), sigma=(2.0, 9)),
    transformsV2.GaussianNoise(),
    transformsV2.JPEG(quality=[50, 100]),  # CPU-bound, cannot run on GPU
    transformsV2.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.4),
    transformsV2.ElasticTransform(alpha=40.0)
])

val_cpu_transforms = transformsV2.Compose([
    transformsV2.ToImage()
])

train_gpu_transforms = lambda t: transformsV2.Compose([
    ## geometric ##
    transformsV2.RandomPerspective(distortion_scale=0.5, p=0.7), # p=0.5 => half of the dataset is affected

    ## photometric ##
    # All the pipeline must be computed on UINT8, conversion at last
    transformsV2.ToDtype(torch.float32, scale=True),
    transformsV2.Normalize(mean=t.mean, std=t.std)
])

val_gpu_transforms = lambda t: transformsV2.Compose([
    # All the pipeline must be computed on UINT8, conversion at last
    transformsV2.ToDtype(torch.float32, scale=True),
    transformsV2.Normalize(mean=t.mean, std=t.std)
])

## Discarded:
# White fill to differ less from the background
# transformsV2.RandomRotation(degrees=(0, 100), fill=255), # let's try but I'm not sure... see https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_rotated_box_transforms.html
# transformsV2.RandomAffine(degrees=(0, 100), fill=255),


