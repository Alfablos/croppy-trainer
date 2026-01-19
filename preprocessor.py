from common import Device
from itertools import chain
from pathlib import Path
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


from utils import resize_img, Precision
from utils import coords_from_segmentation_mask


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
    strict: bool = True,
    precision: Precision | None = None
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """
    
    if csv_path is None and crawler_config is None:
        raise ValueError("At least one of `csv_path` or `crawler_config` must be specified.")
    
    has_coords = False
    
    if csv_path is not None:
        # Read from CSV
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
        # # faster to iterate
        rows = pd.read_csv(csv_path)
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
    
    if compute_coords and not has_coords and not precision:
        raise ValueError("If no mask coordinates are provided a `precision` is needed to compute them. ")
    
    output_dir: Path = Path(output_dir.rstrip('/'))
    
    ## Preflight checks
    # Filesystem is ready
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Destination {output_dir} exists but is a file.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_length = len(rows)

    single_image_size: int = target_h * target_w * 3  # RGB
    total_map_size: int = int(
        data_length * single_image_size * 1.2
    )  # 1.2 is a safety margin
    
    if compute_coords:
        coord_size = 4 * 8 # (8 floats, 32Bit each)
        total_coord_size = int(data_length * coord_size * 1.2)
        total_map_size += total_coord_size
    
    if not has_coords and not compute_coords: # U-Net mode, we're storing the masks!
        mask_size = target_h * target_w * 1 # 1 single channel (B/W) for masks
        total_masks_size = int(mask_size * data_length * 1.2)
        total_map_size += total_masks_size
        
        
    print(f"Allocating {total_map_size / (1024**3):.2f} GB for the lmdb store.")
    
    db_path = str(output_dir) + "/data.lmdb"
    index_path = str(output_dir) + "/index.csv"
    
    if dry_run:
        return
    
    _iter_lmdb_write(
        str(db_path),
        str(index_path),
        total_map_size,
        compute_coords,
        has_coords,
        target_h,
        target_w,
        rows,
        strict,
        precision
    )

    print("Precomputation complete.")





def _iter_lmdb_write(
    db_path: str,
    index_path: str,
    total_map_size: int,
    compute_coords: bool,
    has_coords: bool,
    target_h: int,
    target_w: int,
    rows: list,
    strict: bool,
    precision: Precision
):
    # initialize lmdb at `path`
    env = lmdb.open(db_path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {db_path}.")
    commit_freq = 100
    
    csv_f = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_header = ["index", "path"] + [f"c{k}" for k in range(8)] if compute_coords else ["index", "path"]
    csv_writer.writerow(csv_header)
    csv_index = 0  # NOT updated when images fail to convert
    transaction = env.begin(write=True)
    
    if compute_coords:
        # creates [ "x1", "x2, "x3", ...]
        c_indexes = chain.from_iterable([ [f"x{i}", f"y{i}"] for i in range(1,5)])
        
    try:
        bar = tqdm(total=len(rows), bar_format='{bar}{l_bar}{r_bar}’')
        for i, row in enumerate(rows):
            impath = row["image_path"]
            lpath = row["label_path"]
            bar.set_description(impath)
            
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
                    raise RuntimeError(f"Couldn't resize image at {impath} and strict mode is enforced. Exiting.")
            
            ibytes = pickle.dumps(resized)
            # keys for images: i0, i1, i2, i1 ...
            # keys for labels: l0, l1, l2, l3 ...
            key = str(f"i{csv_index}".encode("ascii"))
            transaction.put(key, ibytes)

            if compute_coords:
                # if coordinates (only checking x1) are present it's ok!
                if 'x1' in row:
                    coords = [row[c] for c in c_indexes]
                    
                elif "label_path" in row:
                    mask_data = cv2.imread(row['label_path'], cv2.IMREAD_GRAYSCALE)
                    coords = coords_from_segmentation_mask(
                        mask_data,
                        precision,
                        device=Device.CPU
                    )
                else:
                    raise ValueError("`compute_coords` is set but no coordinates were found in the csv file nor `label_path` is set in it.")
                
                # Not storing coords in the lmdb
                # transaction.put(f"l{csv_index}".encode("ascii"), pickle.dumps(coords))
                csv_writer.writerow([csv_index, impath, *coords])
                
                
            else:
                # store masks
                print("Storing raw masks in the LMDB file.")
                mpath = row['label_path']
                maskdata = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
                if maskdata is None:
                    if not strict:
                        print(f"Couldn't read image mask at {mpath}. Skipping...")
                        continue
                    else:
                        raise RuntimeError(f"Couldn't read image mask at {mpath} and strict mode is enforced. Exiting.")
                maskdata_resized = resize_img(maskdata, target_h, target_w, interpolation=cv2.INTER_NEAREST)
                if maskdata_resized is None:
                    if not strict:
                        print(f"Couldn't read image mask at {mpath}. Skipping...")
                        continue
                    else:
                        raise RuntimeError(f"Couldn't read image mask at {mpath} and strict mode is enforced. Exiting.")
                transaction.put(f"l{csv_index}", pickle.dumps(maskdata_resized)).encode("ascii")
                csv_writer.writerow([csv_index, impath, lpath])
                
            csv_index += 1

            # i=244 => index[44]
            index[csv_index % commit_freq] = os.path.basename(path)

            # Commit every commit_freq resize operations to save memory
            if ((i + 1) % commit_freq == 0) or (
                i == len(self) - 1
            ):  # every commit_fre iterations and on the last one
                print(f"Saving checkpoint to {checkpoint}")
                transaction.commit()
                transaction = env.begin(write=True)
                pd.DataFrame(index, columns=["path"]).to_csv(
                    checkpoint,
                    mode="a",
                    header=i + 1 == commit_freq,  # only the first time,
                    index=True,
                )

        transaction.put(b"__len__", str(csv_index).encode("ascii"))
        transaction.commit()
        bar.update()
        
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        env.close()
    


    
if __name__ == "__main__":
    precompute(
        output_dir="precomp_test",
        target_h=512,
        target_w=384,
        compute_coords=True,
        csv_path="dataset.csv",
        crawler_config=None,
        dry_run=False,
        verbose=True,
        strict=True
    )