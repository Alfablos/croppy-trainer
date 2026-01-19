from utils import resize_img
import pickle
import cv2
from sympy.physics.quantum.trace import Tr
from crawler import crawl
import os
import csv
from tqdm import tqdm

import lmdb
import numpy as np
import pandas as pd




def precompute(
    output_dir: str,
    # heights of the training images
    target_h: int,
    # weight of the training images
    target_w: int,
    csv_path: str | None = None,
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
    strict: bool = True
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """
    
    if csv_path is None and crawler_config is None:
        raise ValueError("At least one of `csv_path` or `crawler_config` must be specified.")
    
    if csv_path is not None:
        # Read from CSV
        data = pd.read_csv(csv_path)
    else: # crawler_config is not None
        crawler_output = f"./dataset{crawler_config["precision"]}.csv"
        crawler_config["output"] = crawler_output
        
        try:
            crawl(**crawler_config)
        except Exception as e:
            print(e)
            exit(3)
        data = pd.read_csv(crawler_output)
        
    output_dir = output_dir.rstrip('/')
    ## Preflight checks
    # Filesystem is ready
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(f"Destination {output_dir} exists but is a file.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data_length = data.shape[0]

    single_image_size: int = target_h * target_w * 3  # RGB
    total_map_size: int = int(
        data_length * single_image_size * 1.2
    )  # 1.2 is a safety margin
    
    raise Exception("Add the coord size if compute_coord=True")
        
    print(f"Allocating {total_map_size / (1024**3):.2f} GB for the lmdb store.")
    
    db_path = output_dir + "/data.lmdb"
    index_path = output_dir + "/index.csv"
    
    if dry_run:
        exit(0)
    

    # initialize lmdb at `path`
    env = lmdb.open(db_path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {db_path}.")
    commit_freq = 100
    csv_f = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_f)
    if compute_coords:
        csv_writer.writerow(
            ["index", "path"] + [f"c{k}" for k in range(8)]
        )  # index, path, + labels (8 coords of the corners)
    else:
        csv_writer.writerow(["index", "path"])
        
    csv_index = 0  # NOT updated when images fail to convert
    have_coords = data.shape[1] == 10

    transaction = env.begin(write=True)
    try:
        for i, row in enumerate(
            tqdm(data.iterrows(), desc="Saving precomputed examples")
        ):
            impath = row["image_path"]
            lpath = row["label_path"]
            
            imdata = cv2.imread(impath, cv2.IMREAD_COLOR_RGB)
            if imdata is None:
                if not strict:
                    print(f"Couldn't read image at {impath}. Skipping...")
                    continue
                else:
                    print(f"Couldn't read image at {impath} and strict mode is enforced. Exiting.")
                    exit(3)

            resized = resize_img(imdata, target_h, target_w)
            if resized is None:
                if not strict:
                    print(f"Couldn't resize image at {impath}. Skipping...")
                    continue
                else:
                    print(f"Couldn't resize image at {impath} and strict mode is enforced. Exiting.")
                    exit(3)
            # resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            bytes = pickle.dumps(resized)
            key = str(csv_index).encode("ascii")
            transaction.put(key, bytes)

            csv_index += 1

            # i=244 => index[44]
            index[i % commit_freq] = os.path.basename(path)

            # Commit every 100 resize operations to save memory
            if ((i + 1) % commit_freq == 0) or (
                i == len(self) - 1
            ):  # every 100 iterations and on the last one
                print(f"Saving checkpoint to {checkpoint}")
                transaction.commit()
                transaction = env.begin(write=True)
                pd.DataFrame(index, columns=["path"]).to_csv(
                    checkpoint,
                    mode="a",
                    header=i + 1 == commit_freq,  # only the first time,
                    index=True,
                )

        transaction.put(b"__len__", str(len(self)).encode("ascii"))
        transaction.commit()
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        env.close()

    print("Precomputation complete.")
    

    
if __name__ == "__main__":
    precompute(
        output_dir="precomp_test",
        target_h=512,
        target_w=384,
        compute_coords=True,
        csv_path="dataset.csv",
        crawler_config=None,
        dry_run=True,
        verbose=True,
        strict=True
    )