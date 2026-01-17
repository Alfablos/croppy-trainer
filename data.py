import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils


def build_csv(
        root: str,
        output: str,
        images_ext: str,
        labels_ext: str,
        compute_corners=True,
        check_normalization=True,
        verbose=False
):
    if images_ext is None:
        raise ValueError("Please, provide the extension for images. For example `.png` or `_img.png`")
    if labels_ext is None:
        raise ValueError("Please, provide the extension for labels. For example `.png` or `_lbl.png`")

    if not os.path.exists(root):
        raise ValueError(f"Root path {root} does not exist.")

    output = Path(output)
    if output.exists():
        print(f"Output file {output} esists. Refusing to continue.")
        exit(2)

    root = Path(root)
    images = sorted(list(root.glob("**/*" + images_ext)))
    labels = sorted(list(root.glob("**/*" + labels_ext)))
    n_images, n_labels = len(images), len(labels)
    assert n_images == n_labels, "Images and labels differ in number."

    if verbose:
        print(f"Found {len(images)} images with labels.")

    rows = []
    save_chunk_size = 100

    for image, label in tqdm(zip(images, labels, strict=True), total=len(images)):
        try:
            row = {
                'image_path': image,
                'label_path': label
            }

            if compute_corners:
                f = cv2.imread(filename=str(label), flags=cv2.IMREAD_GRAYSCALE)
                mask = np.divide(f, 255.0)
                coords = utils.coords_from_segmentation_mask(mask).numpy()
                fields = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
                for coord_name, value in zip(fields, coords):
                    if value > 1 and check_normalization:
                        print(f"Warning: label {label} has {coord_name} coordinate with a value > 1 ({value}): {coords}\nYou may want to check your normalization algorithm.", file=sys.stderr)

                    row[coord_name] = value

            rows.append(row)

            if len(rows) >= save_chunk_size:
                if verbose:
                    print(f"Saving checkpoint to {output}")
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
        print(f"Done saving labels {'with coordinates' if compute_corners else ''} to {output}")



def save_to_csv(rows: list[dict], dst: str, mode='a'):
    dst_exists = os.path.exists(dst)

    print("Saving to csv")

    if dst_exists and not mode == 'a':
        raise ValueError("Refusing to write to exising CSV file when mode is not 'append'.")

    df = pd.DataFrame(rows, copy=False) # !! Do not reset row yet!
    df.to_csv(
        dst,
        mode=mode,
        header=not dst_exists, # headers only written the first time
        index=False
)




if __name__ == "__main__":
    build_csv(
        root='/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train',
        images_ext='_in.png',
        labels_ext='_gt.png',
        output='./datatest.csv',
        compute_corners=True,
        check_normalization=True,
        verbose=True
    )