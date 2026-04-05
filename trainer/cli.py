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
        for source in sources:
            source_csv = f"{output_dir}/crawl_{source.name}.csv"

            if not os.path.exists(source_csv):
                print(f"[{purpose_key}] Crawling source '{source.name}'...")
                crawl(
                    data_sources=[source],
                    architecture=architecture,
                    output=source_csv,
                    check_normalization=args.check_normalization,
                    verbose=args.verbose,
                    progress=args.progress,
                    limit=args.limit,
                )
            else:
                print(f"[{purpose_key}] Found existing crawl for '{source.name}'. Skipping.")

            precompute(
                architecture=architecture,
                output_dir=output_dir,
                target_h=args.target_height,
                target_w=args.target_width,
                dataset_map_csv=source_csv,
                source_name=source.name,
                dry_run=args.dry_run,
                purpose=Purpose.from_str(purpose_key),
                verbose=args.verbose,
                progress=args.progress,
                strict=args.strict,
                n_workers=args.workers,
                commit_freq=args.commit_frequency,
                compact_store=args.compact_store,
            )

            print(f"\n[{purpose_key}/{source.name}] Precomputation complete.\n")


def _resolve_store_path(store_dir: str, architecture: Architecture, purpose: str, source_name: str, h: int, w: int) -> str:
    """Derive the store path for a given source from the naming convention used by precompute."""
    from common import DEFAULT_STORAGE_CLASS  # avoid circular import at module level
    return f"{store_dir}/{purpose}_data/data_{architecture.value}_{source_name}_{h}x{w}.{DEFAULT_STORAGE_CLASS}"


def _discover_store_dimensions(store_dir: str, architecture: Architecture) -> tuple[int, int]:
    """Read h/w from the first available store in the directory."""
    for arrow_file in sorted(Path(store_dir).rglob(f"data_{architecture.value}_*.arrow")):
        with storage.new_store(str(arrow_file), write=False) as store:
            return int(store.get_metadata("h")), int(store.get_metadata("w"))
    raise FileNotFoundError(f"No stores found for architecture '{architecture}' in {store_dir}")


def run_train(args):
    print("Starting training job...")

    architecture = Architecture.from_str(args.architecture)
    store_dir = args.store_dir.rstrip("/")

    h, w = _discover_store_dimensions(store_dir, architecture)
    print(f"Store dimensions: {h}x{w}")

    # Build per-source training datasets from config
    from torch.utils.data import ConcatDataset

    print("Setting up training dataset...")
    training_datasets = []
    for source in config.DATA_SOURCES["training"]:
        path = _resolve_store_path(store_dir, architecture, "training", source.name, h, w)
        print(f"  Loading {source.name} from {path}")
        ds = SmartDocDataset(
            store_path=path,
            architecture=architecture,
            cpu_transforms=source.train_cpu_transforms,
            precision=Precision.from_str(args.precision),
            limit=args.limit,
        )
        training_datasets.append(ds)
    train_dataset = ConcatDataset(training_datasets) if len(training_datasets) > 1 else training_datasets[0]

    train_dataloader = DataLoader(
        pin_memory=True,
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print("Setting up validation dataset...")
    validation_datasets = []
    for source in config.DATA_SOURCES["validation"]:
        path = _resolve_store_path(store_dir, architecture, "validation", source.name, h, w)
        print(f"  Loading {source.name} from {path}")
        ds = SmartDocDataset(
            store_path=path,
            architecture=architecture,
            cpu_transforms=source.val_cpu_transforms,
            precision=Precision.from_str(args.precision),
            limit=args.limit,
        )
        validation_datasets.append(ds)
    val_dataset = ConcatDataset(validation_datasets) if len(validation_datasets) > 1 else validation_datasets[0]

    val_dataloader = DataLoader(
        pin_memory=True,
        dataset=val_dataset,
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

    # GPU transforms come from the data sources — first source of each purpose
    train_gpu_transforms = config.DATA_SOURCES["training"][0].train_gpu_transforms
    val_gpu_transforms = config.DATA_SOURCES["validation"][0].val_gpu_transforms

    train(
        model=model,
        out_dir=args.output_directory,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        train_gpu_transforms=train_gpu_transforms,
        val_gpu_transforms=val_gpu_transforms,
        epochs=args.epochs,
        train_len=args.limit if args.limit else len(train_dataset),
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
