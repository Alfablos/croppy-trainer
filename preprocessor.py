from pandas.io.xml import preprocess_data
from common import Device
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


from utils import resize_img, Precision, assert_never, coords_from_segmentation_mask
from architecture import Architecture


def precompute(
    architecture: Architecture,
    db_output_dir: str,
    # heights of the training images
    target_h: int,
    # weight of the training images
    target_w: int,
    dataset_map_csv: str | None = None,
    crawler_config: dict | None = None,
    # Every how many iterations data is written to disk
    commit_freq: int = 100,
    # No actual computation
    dry_run: bool = False,
    verbose: bool = False,
    # Whether the output should contain corners coords
    # If true they'll be computed if the csv does not contain them alread
    # If csv_path is none so the file will be computed and will
    # contain the coordinates depending on this value
    compute_coords: bool = True,
    strict: bool = True,
    precision: Precision | None = None
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """
    args = locals()
    
    if dataset_map_csv is None and crawler_config is None:
        # the user needs crawling first to generate the csv but provides no crawler config
        raise ValueError("At least one of `csv_path` or `crawler_config` must be specified.")
    
    err  = architecture.validate_preprocessor_config(args)
    if err:
        raise ValueError(f"Invalid configuration for a {Architecture.value}: {err}")
    
    if dataset_map_csv is not None:
        # Read from CSV
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
        # # faster to iterate
        rows = pd.read_csv(dataset_map_csv)
    else: # crawler_config is not None
        crawler_output = f"./dataset{crawler_config["precision"]}.csv"
        crawler_config["output"] = crawler_output
        
        try:
            print(f"Crawling data at {crawler_config.get("root")}")
            crawler_config["compute_corners"] = compute_coords
            crawl(**crawler_config)
        except Exception as e:
            print(e)
            exit(3)
        rows = pd.read_csv(crawler_output)
    
    rows = rows.to_dict("records")
    has_coords = 'x1' in rows[0]
    
    # resnet only, compute coords is not compatible with Architecture.UNET
    if architecture == Architecture.RESNET and not has_coords and not precision:
        raise ValueError("If no mask coordinates are provided a `precision` is needed to compute them. ")
    
    db_output_dir: Path = Path(db_output_dir.rstrip('/'))
    
    ## Preflight checks
    # Filesystem is ready
    if db_output_dir.exists() and not db_output_dir.is_dir():
        raise FileExistsError(f"Destination {db_output_dir} exists but is a file.")
    db_output_dir.mkdir(parents=True, exist_ok=True)
    
    data_length = len(rows)

    total_map_size = architecture.preprocessor_db_map_size(
        data_length,
        target_h,
        target_w
    )
    
    print(f"Allocating {total_map_size / (1024**3):.2f} GB for the lmdb store.")
    
    db_path = str(db_output_dir) + "/data.lmdb"
    index_path = str(db_output_dir) + "/index.csv"
    
    if dry_run:
        return
    
    env = lmdb.open(db_path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {db_path}.")
    csv_index_file = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_index_file)
    csv_header = architecture.get_csv_header()
    db_index = 0  # NOT updated when images fail to convert (if not strict)
    transaction = env.begin(write=True)
    
    transform = architecture.get_transform_logic()
    
    try:
        bar = tqdm(total=len(rows), bar_format='{bar}{l_bar}{r_bar}’')
        for i, row in enumerate(rows):
            impath = row["image_path"]
            lpath = row["label_path"]
            bar.set_description(impath)
            
            img_resized, label = transform(
                row,
                target_h,
                target_w,
                precision
            )
            
            if architecture == Architecture.RESNET:
                csv_writer.writerow([db_index, impath, *label])
            elif architecture == Architecture.UNET:
                csv_writer.writerow([db_index, impath, lpath])
            
            # Put image
            ibytes = pickle.dumps(img_resized)
            # keys for images: i0, i1, i2, i1 ...
            # keys for labels: l0, l1, l2, l3 ...
            ikey = f"i{db_index}".encode("ascii")
            transaction.put(ikey, ibytes)
            
            # Put label
            lbytes = pickle.dumps(label)
            lkey = f"l{db_index}".encode("ascii")
            transaction.put(lkey, lbytes)
            
            # Commit every commit_freq resize operations to save memory
            if ((i + 1) % commit_freq == 0) or (
                i == len(rows) - 1
            ):  # every commit_fre iterations and on the last one
                transaction.commit()
                transaction = env.begin(write=True)
            
            bar.update(1)
            
            db_index += 1
        
        # Watch out: the index refers to image-label pairs!
        transaction.put(b"__len__", str(db_index).encode("ascii"))
        transaction.commit()
        
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        env.close()
    
    print("Precomputation complete.")






    
if __name__ == "__main__":
    precompute(
        db_output_dir="precomp_test",
        target_h=512,
        target_w=384,
        compute_coords=True,
        dataset_map_csv="dataset.csv",
        crawler_config=None,
        dry_run=False,
        verbose=True,
        strict=True,
        architecture=Architecture.RESNET
    )