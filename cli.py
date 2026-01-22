from jinja2.nodes import FromImport
from multiprocessing import cpu_count
import argparse
from torch.nn import L1Loss
from train import train
from torch.utils.data import DataLoader
from data import SmartDocDataset
import torch
from architecture import Architecture
from preprocessor import precompute
from common import Precision, Device
from pathlib import Path
from crawler import crawl

import torchvision.models as visionmodels
from torchvision.transforms import v2 as transformsV2

def version(module):
    print(f'{module.__name__}=={module.__version__}')

def dependencies(_args):
    import numpy
    version(numpy)
    import pandas
    version(pandas)
    import pytest
    version(pytest)
    import torch
    version(torch)
    import torchvision
    version(torchvision)
    import tensorboard
    version(tensorboard)
    import PIL
    version(PIL)
    import cv2
    version(cv2)
    import matplotlib
    version(matplotlib)
    import tqdm
    version(tqdm)
    import lmdb
    version(lmdb)
    

# complete this function
def run_crawl(args):
    crawl(
        root=Path(args.data_root),
        images_ext=args.image_extension,
        labels_ext=args.label_extension,
        output=args.output,
        precision=Precision.from_str(args.precision),
        compute_corners=args.compute_corners,
        check_normalization=args.check_normalization,
        verbose=args.verbose,
    )

def run_precompute(args):
    if not args.data_map:
        crawler_config = {
            "root": Path(args.data_root),
            "images_ext": args.image_extension,
            "labels_ext": args.label_extension,
            "precision": Precision.from_str(args.precision),
            "compute_corners": args.compute_corners,
            "check_normalization": args.check_normalization,
            "verbose": args.verbose
        }
    else:
        crawler_config = None

    kwargs = {}
    if args.commit_frequency:
        kwargs["commit_freq"] = int(args.commit_frequency)
    if args.workers:
        kwargs["n_workers"] = int(args.workers)

    precompute(
        architecture=Architecture.from_str(args.architecture),
        db_output_dir=args.output_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        dataset_map_csv=args.data_map,
        crawler_config=crawler_config,
        dry_run=args.dry_run,
        verbose=args.verbose,
        compute_corners=args.compute_corners,
        strict=args.strict,
        precision=Precision.from_str(args.precision)
    )
    
    
def run_train(args):
    weights = visionmodels.ResNet18_Weights.DEFAULT

    t = weights.transforms()
    normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
    train_transform = transformsV2.Compose(
        [
            transformsV2.ToImage(),
            transformsV2.JPEG(quality=[50, 100]),
            transformsV2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )
    
    resnet_train_ds = SmartDocDataset(
        lmdb_path=args.lmdb_path,
        architecture=Architecture.from_str(args.architecture),
        precision=Precision.from_str(args.precision),
        image_transform=train_transform,
        label_transform=None,
        limit=args.limit
    )

    dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    
    train(
        train_dataloader=dataloader,
        mode_weights=weights,
        device=Device.from_str(args.device),
        dropout=args.dropout,
        loss_function=L1Loss(),
        learning_rate=args.learning_rate,
        # epochs=100,
        epochs=args.epochs,
        verbose=args.verbose,
        out_file=args.output_file,
    )
    
    
    