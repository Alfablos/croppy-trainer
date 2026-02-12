import os
from pathlib import Path
from time import sleep

import cv2
from numpy.typing import NDArray
from torch.utils.data import DataLoader

import storage
import utils
from architecture import Architecture
from common import DEFAULT_WEIGHTS
from common import Precision, Device, Purpose
from crawler import crawl
from data import SmartDocDataset
from inference import predict, get_image_points, draw_box
from loss import loss_from_str
from preprocessor import precompute
from train import train, CroppyNet


def run_crawl(args):
    crawl(
        root=Path(args.data_root),
        images_ext=args.image_extension,
        labels_ext=args.label_extension,
        output=args.output,
        compute_corners=args.compute_corners,
        coords_scale_percentage=float(args.corners_recess_percentage),
        check_normalization=args.check_normalization,
        verbose=args.verbose,
        progress=args.progress,
        limit=args.limit,
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
            root=Path(args.data_root),
            output=data_map,
            images_ext=args.image_extension,
            labels_ext=args.label_extension,
            compute_corners=args.compute_corners,
            coords_scale_percentage=float(args.corners_recess_percentage),
            check_normalization=args.check_normalization,
            verbose=args.verbose,
            progress=args.progress,
            limit=args.limit,
        )
    else:
        print(f"Found data map file {data_map}. Skipping crawler.")

    precompute(
        architecture=Architecture.from_str(args.architecture),
        output_dir=args.output_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        dataset_map_csv=crawler_output,
        dry_run=args.dry_run,
        purpose=Purpose.from_str(args.purpose),
        verbose=args.verbose,
        progress=args.progress,
        compute_corners=args.compute_corners,
        coords_scale_percentage=float(args.corners_recess_percentage),
        strict=args.strict,
        n_workers=args.workers,
        commit_freq=args.commit_frequency,
        compact_store=args.compact_store,
    )


def run_train(args):
    print("Starting training job...")
    weights = DEFAULT_WEIGHTS

    # Retrieve height and width from the LMDB store
    print(f"Opening store at {args.store_path}")
    with storage.new_store(args.store_path, write=False) as store:
        h = int(store.get_metadata("h"))
        w = int(store.get_metadata("w"))

    print(f"Setting up training dataset...")

    resnet_train_ds = SmartDocDataset(
        store_path=args.store_path,
        architecture=Architecture.from_str(args.architecture),
        train=True,
        precision=Precision.from_str(args.precision),
        limit=args.limit,
    )

    train_dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(f"Setting up validation dataset...")
    resnet_val_ds = SmartDocDataset(
        store_path=args.validation_store_path,
        architecture=Architecture.from_str(args.architecture),
        train=args.hard_validation,
        precision=Precision.from_str(args.precision),
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
        loss_fn=loss_from_str(args.loss_function),
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
        hard_validation=args.hard_validation,
        verbose=args.verbose,
        progress=args.progress,
        with_tensorboard=args.enable_tensorboard,
        debug=int(args.debug) if args.debug is not None else None,
        checkpoint=args.checkpoint,
    )


def run_predict(args):
    config = utils.load_checkpoint(args.config, train=False)
    model = CroppyNet.from_trained_config(config, Device.from_str(args.device))

    # load image and convert to RGB from BGR
    image = cv2.imread(args.path, cv2.IMREAD_COLOR)
    image: NDArray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image, original_shape = Architecture.resize_image(
        args.path,
        h=model.images_height,
        w=model.images_width,
        color=True,
        interpolation=cv2.INTER_AREA,
        allow_padding=True,
    )

    norm_coords = predict(
        image=resized_image, model=model, device=Device.from_str(args.device)
    )
    actual_coords = get_image_points(
        image.shape,  # NOT RESIZED!
        norm_coords,
    )

    draw_box(actual_coords, image)

    outpath = args.output
    cv2.imwrite(outpath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(f"Image saved to: {outpath}")
