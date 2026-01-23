from pyexpat import model
from sympy.functions.special.tests.test_error_functions import w
from inference import predict
from time import sleep
import os
from multiprocessing import cpu_count
import argparse
from torch.nn import L1Loss
from train import train, CroppyNet
from torch.utils.data import DataLoader
from data import SmartDocDataset
import torch
from architecture import Architecture
from preprocessor import precompute
from common import Precision, Device, Purpose
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
    crawler_output = f"./dataset_{str(args.architecture)}_{str(args.precision).lower()}_{args.purpose}.csv"
    if not args.data_map:
        data_map = crawler_output
    else:
        data_map = args.data_map
    
    if not os.path.exists(data_map):
        print(f"Crawler output not found at {crawler_output}, data needs to be crawled first.")
        print(f"If you have already crawled your data root rename the output file to `{crawler_output}`.")
        print(f"waiting 5 seconds before starting to crawl, interrupt now if you don't wish to continue.")
        sleep(5)
        crawler_config = {
            "root": Path(args.data_root),
            "output": crawler_output,
            "images_ext": args.image_extension,
            "labels_ext": args.label_extension,
            "precision": Precision.from_str(args.precision),
            "compute_corners": args.compute_corners,
            "check_normalization": args.check_normalization,
            "verbose": args.verbose
        }
        crawl(**crawler_config)
        

    precompute(
        architecture=Architecture.from_str(args.architecture),
        output_dir=args.output_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        dataset_map_csv=crawler_output,
        dry_run=args.dry_run,
        purpose=Purpose.from_str(args.purpose),
        verbose=args.verbose,
        compute_corners=args.compute_corners,
        strict=args.strict,
        precision=Precision.from_str(args.precision),
        n_workers=args.workers,
        commit_freq=args.commit_frequency
    )
    
    
def run_train(args):
    weights = visionmodels.ResNet18_Weights.DEFAULT

    t = weights.transforms()
    normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
    train_t = [
                transformsV2.ToImage(),
                transformsV2.JPEG(quality=[50, 100]),
                transformsV2.ToDtype(dtype=torch.float32, scale=True),
                normalize,
            ]
    train_transforms = transformsV2.Compose(train_t)
    
    
    resnet_train_ds = SmartDocDataset(
        lmdb_path=args.lmdb_path,
        architecture=Architecture.from_str(args.architecture),
        precision=Precision.from_str(args.precision),
        image_transforms=train_transforms,
        label_transforms=None,
        limit=args.limit
    )

    train_dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    
    if args.validation_lmdb_path:
        val_t = [
            transformsV2.ToImage(),
            transformsV2.ToDtype(dtype=torch.float32, scale=True)
        ]
        val_transforms = transformsV2.Compose(val_t)
        resnet_val_ds = SmartDocDataset(
            lmdb_path=args.validation_lmdb_path,
            architecture=Architecture.from_str(args.architecture),
            precision=Precision.from_str(args.precision),
            image_transforms=val_transforms,
            label_transforms=None,
            limit=args.limit
        )
    
        val_dataloader = DataLoader(
            pin_memory=True,  # Using CUDA
            dataset=resnet_val_ds,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    
    train(
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        mode_weights=weights,
        device=Device.from_str(args.device),
        dropout=args.dropout,
        loss_function=L1Loss(),
        learning_rate=args.learning_rate,
        # epochs=100,
        epochs=args.epochs,
        tensorboard_logdir=args.tensorboard_logdir,
        verbose=args.verbose,
        progress=args.progress,
        out_file=args.output_file,
        with_tensorboard=args.enable_tensorboard
    )
    
    
def run_predict(args):
    model = CroppyNet(visionmodels.ResNet18_Weights.DEFAULT)
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    model = model.to(Device.CUDA.value)
    result = predict(
        img_path=args.path,
        architecture=Architecture.from_str(args.architecture),
        h=args.height,
        w=args.width,
        model=model,
        device=Device.CUDA
    )
    print(result)
    