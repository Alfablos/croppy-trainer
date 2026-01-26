import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


from common import Precision, Purpose
import utils
from utils import Device


def crawl(
    root: Path,
    output: str,
    images_ext: str,
    labels_ext: str,
    compute_corners=True,
    check_normalization=True,
    verbose=False,
):
    """
    Crawls a directory structure to find and pair image files with their corresponding label files, optionally computes normalized document corner coordinates from segmentation masks, and saves the results to a CSV file.

    The function implements checkpointing by saving intermediate results every 100 rows, and handles KeyboardInterrupt gracefully by saving any remaining processed rows before exiting.

    Args:
        root (Path): Root directory path to search for image and label files. Must exist.
        output (str): Path to the output CSV file where results will be saved. The file must not already exist to  accidental overwrites.
        images_ext (str): Extension pattern to match image files (e.g., `.png`, `_img.png`).
            This is used with glob patterns to find images recursively.
        labels_ext (str): Extension pattern to match label/segmentation mask files
            (e.g., `.png`, `_lbl.png`). Labels must match images in number and be pairable by
            their sorted order.
        precision (Precision): Target numerical precision for future training.
            Determines the dtype used when loading segmentation masks and affects normalization
            behavior (normalization for Float32/Float16, none otherwise). Options are FP32, FP16, or UINT8.
        compute_corners (bool, optional): If True, computes normalized corner coordinates (x1, y1,
            x2, y2, x3, y3, x4, y4) from segmentation masks using `utils.coords_from_segmentation_mask`.
            If False, only saves image and label paths without coordinates. Defaults to True.
        check_normalization (bool, optional): If True and `precision` is not UINT8, warns when
            computed coordinates have values > 1, indicating potential normalization issues in the
            preprocessing pipeline. Defaults to True.
        verbose (bool, optional): If True, prints progress information and uses tqdm to show a
            progress bar. Defaults to False.

    Raises:
        ValueError: If `images_ext` is None.
        ValueError: If `labels_ext` is None.
        ValueError: If the root directory does not exist.
        AssertionError: If the number of images and labels found do not match.
        SystemExit: Exits with code 2 if the output file already exists.
        SystemExit: Exits with code 0 on KeyboardInterrupt after saving remaining rows.

    Side Effects:
        - Creates/appends to a CSV file at `output` path containing columns:
            * image_path (Path): Path to the image file
            * label_path (Path): Path to the label file
            * x1, y1, x2, y2, x3, y3, x4, y4 (float, optional): Normalized corner coordinates
              if `compute_corners` is True
        - Prints progress and warning messages to stdout/stderr if `verbose` is True
        - Loads segmentation masks from disk for coordinate computation

    Notes:
        - Images and labels are paired by their sorted order using `strict=True` in zip.
        - Segmentation masks are loaded as grayscale and normalized to [0, 1] range for FP32/FP16,
          or kept as raw uint8 values for UINT8 precision.
        - Corner coordinates are computed assuming rectangular/squared documents that may be
          rotated. The coordinates represent the four corners in normalized [0, 1] space.
        - The function saves intermediate results every 100 rows to provide checkpointing in
          case of interruptions.
    """
    if images_ext is None:
        raise ValueError(
            "Please, provide the extension for images. For example `.png` or `_img.png`"
        )
    if labels_ext is None:
        raise ValueError(
            "Please, provide the extension for labels. For example `.png` or `_lbl.png`"
        )

    if not root.exists():
        raise ValueError(f"Root path {root} does not exist.")

    if os.path.exists(output):
        print(f"Output file {output} esists. Refusing to continue.")
        exit(2)

    images = sorted(list(root.glob("**/*" + images_ext)))
    labels = sorted(list(root.glob("**/*" + labels_ext)))
    n_images, n_labels = len(images), len(labels)
    assert n_images == n_labels, "Images and labels differ in number."

    if verbose:
        print(f"Found {len(images)} images with labels.")

    rows = []
    save_chunk_size = 100

    if verbose:
        progress = tqdm(total=len(images), desc="Pairing examples and labels")

    output_p = Path(output)
    if not output_p.parent.exists():
        output_p.parent.mkdir(parents=True)

    for image, label in zip(images, labels, strict=True):
        try:
            row = {"image_path": image, "label_path": label}

            if compute_corners:
                mask = cv2.imread(filename=str(label), flags=cv2.IMREAD_GRAYSCALE)

                coords = utils.coords_from_segmentation_mask(mask, device=Device.CPU)
                fields = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
                for coord_name, value in zip(fields, coords):
                    row[coord_name] = value

            rows.append(row)

            if verbose:
                progress.update(1)

            if len(rows) >= save_chunk_size:
                save_to_csv(rows, str(output))
                rows = []

        except KeyboardInterrupt:
            if rows:
                print("Saving remaining rows after user interruption...")
                save_to_csv(rows, str(output))
                print("Done.")
                exit(0)
    if rows:
        save_to_csv(rows, str(output))

    if verbose:
        print(
            f"Done saving labels {'with coordinates' if compute_corners else ''} to {output}"
        )


def save_to_csv(rows: list[dict], dst: str, mode="a"):
    dst_exists = os.path.exists(dst)

    if dst_exists and not mode == "a":
        raise ValueError(
            "Refusing to write to exising CSV file when mode is not 'append'."
        )

    df = pd.DataFrame(rows, copy=False)  # !! Do not reset row yet!
    df.to_csv(
        dst,
        mode=mode,
        header=not dst_exists,  # headers only written the first time
        index=False,
    )
