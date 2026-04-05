import os
from pathlib import Path
from time import sleep

import cv2
from numpy.typing import NDArray
from torch.utils.data import DataLoader

import storage
import utils
from architecture import Architecture
from common import Precision, Device, Purpose
from crawler import crawl
from data import SmartDocDataset
from inference import predict, get_image_points, draw_box
from loss import loss_from_str
from preprocessor import precompute
from train import train, CroppyNet

import config
from storage import merge_arrow_stores


def run_merge_stores(args):
    print(f"Merging {len(args.stores)} stores...")
    merge_arrow_stores(args.stores, args.output)


def run_crawl(args):
    # Standalone crawl: flatten all sources across all purposes
    all_sources = [s for sources in config.DATA_SOURCES.values() for s in sources]
    crawl(
        data_sources=all_sources,
        architecture=Architecture.from_str(args.architecture),
        output=args.output,
        check_normalization=args.check_normalization,
        verbose=args.verbose,
        progress=args.progress,
        limit=args.limit,
    )


def run_precompute(args):
    architecture = Architecture.from_str(args.architecture)
    output_dir = args.output_dir.rstrip("/")

    for purpose_key, sources in config.DATA_SOURCES.items():
        combined_csv = f"{output_dir}/dataset_{str(architecture)}_{purpose_key}.csv"

        # Crawl each source into its own CSV (skip if already exists):
        # prevents having to start all from scratch in case of an error
        # in later sources
        source_csvs = []
        needs_merge = False
        for source in sources:
            source_csv = f"{output_dir}/crawl_{source.name}.csv"
            source_csvs.append(source_csv)

            if not os.path.exists(source_csv):
                print(
                    f"[{purpose_key}] Crawling source '{source.name}'..."
                )
                crawl(
                    data_sources=[source],
                    architecture=architecture,
                    output=source_csv,
                    check_normalization=args.check_normalization,
                    verbose=args.verbose,
                    progress=args.progress,
                    limit=args.limit,
                )
                needs_merge = True
            else:
                print(f"[{purpose_key}] Found existing crawl for '{source.name}'. Skipping.")

        # Merge per-source CSVs into a combined CSV for precompute
        if needs_merge or not os.path.exists(combined_csv):
            import pandas as pd
            frames = [pd.read_csv(csv) for csv in source_csvs if os.path.exists(csv)]
            merged = pd.concat(frames, ignore_index=True)
            merged.to_csv(combined_csv, index=False)
            print(f"[{purpose_key}] Merged {len(frames)} source(s) into {combined_csv} ({len(merged)} rows).")

        precompute(
            architecture=architecture,
            output_dir=output_dir,
            target_h=args.target_height,
            target_w=args.target_width,
            dataset_map_csv=combined_csv,
            dry_run=args.dry_run,
            purpose=Purpose.from_str(purpose_key),
            verbose=args.verbose,
            progress=args.progress,
            strict=args.strict,
            n_workers=args.workers,
            commit_freq=args.commit_frequency,
            compact_store=args.compact_store,
        )

        print(f"\n[{purpose_key}] Precomputation complete.\n")


def run_train(args):
    print("Starting training job...")

    # Retrieve height and width from the store
    print(f"Opening store at {args.training_store_path}")
    with storage.new_store(args.training_store_path, write=False) as store:
        h = int(store.get_metadata("h"))
        w = int(store.get_metadata("w"))

    print(f"Setting up training dataset...")

    resnet_train_ds = SmartDocDataset(
        store_path=args.training_store_path,
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

    resume_checkpoint = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_checkpoint = utils.load_checkpoint(args.resume)
        model = CroppyNet.from_trained_config(resume_checkpoint, Device.from_str(args.device))
    else:
        model = CroppyNet(
            architecture=Architecture.from_str(args.architecture),
            loss_fn=loss_from_str(args.loss_function),
            images_height=h,
            images_width=w,
            target_device=Device.from_str(args.device),
            learning_rate=args.learning_rate,
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
        resume_from=resume_checkpoint,
    )


def run_predict(args):
    checkpoint = utils.load_checkpoint(args.checkpoint)
    device = Device.from_str(args.device)
    model = CroppyNet.from_trained_config(checkpoint, device)

    # Read original-size image for drawing the result
    original_image = cv2.imread(args.path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"Could not read image at {args.path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Resize + pad to model input dimensions for inference
    resized_image, _ = Architecture.resize_image(
        args.path,
        h=model.images_height,
        w=model.images_width,
        color=True,
        interpolation=cv2.INTER_AREA,
        allow_padding=True,
    )

    norm_coords = predict(image=resized_image, model=model, device=device)

    # Denormalize to original image pixel coordinates
    actual_coords = get_image_points(original_image.shape, norm_coords)

    draw_box(actual_coords, original_image)

    outpath = args.output
    cv2.imwrite(outpath, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {outpath}")
