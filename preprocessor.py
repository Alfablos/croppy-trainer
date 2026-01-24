from time import sleep
from numpy.typing import NDArray
from torch.multiprocessing import cpu_count
import multiprocessing
from functools import partial
from typing import Callable
from pandas.io.xml import preprocess_data
from common import Purpose
from itertools import chain
from pathlib import Path
import pickle
import cv2
from crawler import crawl
import os
import csv
from tqdm import tqdm

import lmdb
import numpy as np
import pandas as pd


from common import Precision
from utils import resize_img, assert_never, coords_from_segmentation_mask
from architecture import Architecture, ProcessResult


def worker(
    row: dict,
    transform_fn: Callable[[dict, int, int], ProcessResult],
    target_h: int,
    target_w: int,
    strict: bool,
) -> tuple[dict, NDArray, NDArray] | None:
    try:
        result = transform_fn(row, target_h, target_w)
        img, label = result.image, result.label
        return row, img, label
    except Exception as e:
        if strict:
            raise e
        print(f"Skipping {row['image_path']} due to an error:", e)
        return None


def precompute(
    architecture: Architecture,
    purpose: Purpose,
    output_dir: str,
    # heights of the training images
    target_h: int,
    # weight of the training images
    target_w: int,
    dataset_map_csv: str,
    # Every how many iterations data is written to disk
    commit_freq: int = 100,
    # No actual computation
    dry_run: bool = False,
    verbose: bool = False,
    # Whether the output should contain corners coords
    # If true they'll be computed if the csv does not contain them alread
    # If csv_path is none so the file will be computed and will
    # contain the coordinates depending on this value
    compute_corners: bool = True,
    strict: bool = True,
    n_workers: int = int(cpu_count() / 2),
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """

    args = locals()

    err = architecture.find_preprocessor_misconfig(args)
    if err:
        raise ValueError(f"Invalid configuration for a {architecture.value}: {err}")

    if output_dir in [".", "./"]:
        output_dir = f"{purpose}_data"
    else:
        output_dir = output_dir.rstrip("/") +'/' + f"{purpose}_data"

    rows = pd.read_csv(dataset_map_csv).to_dict("records")
    has_coords = "x1" in rows[0]


    output_dir: Path = Path(output_dir)

    ## Preflight checks
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Destination {output_dir} exists but is a file.")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_length = len(rows)

    total_map_size = architecture.preprocessor_db_map_size(
        data_length, target_h, target_w
    )

    print(f"Allocating {total_map_size / (1024**3):.2f} GB for the lmdb store.")

    db_path = str(output_dir) + f"/data_{architecture.value}_{str(purpose)}_{data_length}x{target_h}x{target_w}.lmdb"
    if os.path.exists(db_path):
        raise FileExistsError(db_path)
    index_path = str(output_dir) + f"/index_{architecture.value}_{str(purpose)}_{data_length}x{target_h}x{target_w}.csv"
    if os.path.exists(index_path):
        raise FileExistsError(index_path)

    if dry_run:
        return
    else:
        print("Waiting 5 seconds before starting, presso Ctrl + c to interrupt...")
        sleep(5)

    env = lmdb.open(db_path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {db_path}.")
    csv_index_file = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_index_file)
    csv_header = architecture.get_csv_header()
    db_index = 0  # NOT updated when images fail to convert (if not strict)
    transaction = env.begin(write=True)

    transform = architecture.get_transform_logic()

    worker_f = partial(
        worker,
        transform_fn=transform,
        target_h=target_h,
        target_w=target_w,
        strict=strict,
    )

    with multiprocessing.Pool(n_workers) as pool:
        result_iter = pool.imap(
            worker_f, rows, chunksize=10
        )  # trying to preserve order, not using imap_unordered

        if verbose:
            bar = tqdm(total=len(rows), bar_format="{bar}{l_bar}{r_bar}’")

        try:
            for result in result_iter:
                if not result:
                    if verbose:
                        bar.update(1)
                    continue

                row, img, label = result
                ipath = row["image_path"]
                lpath = row["label_path"]

                if architecture == Architecture.RESNET:
                    csv_writer.writerow([db_index, ipath, *label])
                elif architecture == Architecture.UNET:
                    csv_writer.writerow([db_index, ipath, lpath])

                # Put image
                ibytes = pickle.dumps(img)
                # keys for images: i0, i1, i2, i1 ...
                # keys for labels: l0, l1, l2, l3 ...
                ikey = f"i{db_index}".encode("ascii")
                transaction.put(ikey, ibytes)

                # Put label
                lbytes = pickle.dumps(label)
                lkey = f"l{db_index}".encode("ascii")
                transaction.put(lkey, lbytes)

                # Commit every commit_freq resize operations to save memory
                if ((db_index + 1) % commit_freq == 0) or (
                    db_index == len(rows) - 1
                ):  # every commit_fre iterations and on the last one
                    transaction.commit()
                    env.sync()  # forces filesystem synchronization
                    transaction = env.begin(write=True)

                if verbose:
                    bar.update(1)
                db_index += 1

        except Exception as e:
            print(f"Error: {e}")
            raise e
        finally:
            transaction.put("__len__".encode("ascii"), db_index.to_bytes(64, "big"))
            transaction.put("h".encode("ascii"), target_h.to_bytes(64, "big"))
            transaction.put("w".encode("ascii"), target_w.to_bytes(64, "big"))
            transaction.commit()
            env.sync()
            env.close()

    print("Precomputation complete.")


if __name__ == "__main__":
    precompute(
        output_dir="precomp_test",
        target_h=512,
        target_w=384,
        compute_corners=True,
        dataset_map_csv="dataset.csv",
        dry_run=False,
        verbose=True,
        strict=True,
        architecture=Architecture.RESNET,
        n_workers=13,
        purpose=Purpose.TRAIN,
    )
