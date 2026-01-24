import cv2
from numpy.typing import NDArray
import lmdb
from torch.distributed.checkpoint.planner import WriteItem
import json
from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from pyexpat import model
from sympy.functions.special.tests.test_error_functions import w
from inference import predict, get_image_points, draw_box
from time import sleep
import os
from multiprocessing import cpu_count
import argparse
from torch.nn import L1Loss
from train import train, CroppyNet
from torch.utils.data import DataLoader
from data import SmartDocDataset, get_transforms
import torch
from architecture import Architecture
from preprocessor import precompute
from common import Precision, Device, Purpose
from pathlib import Path
from crawler import crawl

import torchvision.models as visionmodels
from torchvision.transforms import v2 as transformsV2
from train import DEFAULT_WEIGHTS


def version(module):
    import re
    print(f"{module.__name__}==" + re.sub(r"\+.*", "", module.__version__))


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
        compute_corners=args.compute_corners,
        check_normalization=args.check_normalization,
        verbose=args.verbose,
    )


def run_precompute(args):
    crawler_output = f"{args.output_dir.rstrip('/')}/dataset_{str(args.architecture)}_{args.purpose}.csv"
    if not args.data_map:
        data_map = crawler_output
    else:
        data_map = args.data_map

    if not os.path.exists(data_map):
        print(
            f"Crawler output not found at {crawler_output}, data needs to be crawled first."
        )
        print(
            f"If you have already crawled your data root rename the output file to `{crawler_output}`."
        )
        print(
            f"waiting 5 seconds before starting to crawl, interrupt now if you don't wish to continue."
        )
        sleep(5)
        crawl(
            root = Path(args.data_root),
            output = data_map,
            images_ext = args.image_extension,
            labels_ext = args.label_extension,
            compute_corners = args.compute_corners,
            check_normalization = args.check_normalization,
            verbose = args.verbose,
        )

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
        n_workers=args.workers,
        commit_freq=args.commit_frequency,
    )


def run_train(args):
    weights = DEFAULT_WEIGHTS
    
    # Retrieve height and width from the LMDB store
    env = lmdb.open(args.lmdb_path, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as t:
        h = int.from_bytes(t.get("h".encode("ascii")), "big")
        w = int.from_bytes(t.get("w".encode("ascii"), "big"))
        
    

    train_transforms = get_transforms(
        weights=weights, precision=Precision.from_str(args.precision), train=True
    )

    resnet_train_ds = SmartDocDataset(
        lmdb_path=args.lmdb_path,
        architecture=Architecture.from_str(args.architecture),
        precision=Precision.from_str(args.precision),
        image_transforms=train_transforms,
        label_transforms=None,
        limit=args.limit,
    )

    train_dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    if args.validation_lmdb_path:
        val_transforms = get_transforms(
            weights=weights, precision=Precision.from_str(args.precision), train=False
        )
        resnet_val_ds = SmartDocDataset(
            lmdb_path=args.validation_lmdb_path,
            architecture=Architecture.from_str(args.architecture),
            precision=Precision.from_str(args.precision),
            image_transforms=val_transforms,
            label_transforms=None,
            limit=args.limit,
        )

        val_dataloader = DataLoader(
            pin_memory=True,  # Using CUDA
            dataset=resnet_val_ds,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    model = CroppyNet(
        weights=weights,
        architecture=Architecture.from_str(args.architecture),
        precision=args.precision,
        loss_fn=L1Loss(),
        images_height=h,
        images_width=w,
        target_device=Device.from_str(args.device),
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    train(
        model=model,
        out_dir=args.output_directory,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        epochs=args.epochs,
        train_len=args.limit if args.limit else len(resnet_train_ds),
        verbose=args.verbose,
        progress=args.progress,
        with_tensorboard=args.enable_tensorboard,
    )


def run_predict(args):
    with open(args.config, 'r') as c:
        config = json.loads(c.read())
        device = args.device
        model = CroppyNet.from_trained_config(config, Device.from_str(device))

    # load image and convert to RGB from BGR
    image = cv2.imread(args.path, cv2.IMREAD_COLOR)
    image: NDArray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized_image: NDArray = Architecture.resize_image(
        args.path, h=model.images_height, w=model.images_width, color=True, resize=True, interpolation=cv2.INTER_AREA
    )
    
    norm_coords = predict(
        image=resized_image,
        model=model,
        device=Device.from_str(args.device)
    )
    actual_coords = get_image_points(
        image.shape, # NOT RESIZED!
        norm_coords
    )
    
    draw_box(actual_coords, image)
    
    outpath = args.path + '_pred.jpg'
    cv2.imwrite(outpath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print(f"Image saved to: {outpath}")

