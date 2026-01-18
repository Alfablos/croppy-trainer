import os
import csv
from tqdm import tqdm

import lmdb
import numpy as np
import pandas as pd




def precompute(
    csv_path: str | None,
    output_path: str,
    # heights of the training images
    target_h: int,
    # weight of the training images
    target_w: int,
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
    if not path.endswith(".lmdb"):
        print(f"Warning: saving a LMDB file without the `.lmdb` extension.")
    if os.path.exists(path):
        raise FileExistsError(
            f"The path '{path}' already exists. Refusing to continue."
        )
    single_image_size: int = target_h * target_w * 3  # RGB
    total_map_size: int = int(
        len(self) * single_image_size * 1.2
    )  # 1.2 is a safety margin
    print(f"Allocating {total_map_size / (1024**3)} GB for the lmdb store.")

    # initialize lmdb at `path`
    env = lmdb.open(path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {path}.")
    commit_freq = 100
    csv_f = open(f"{path}_index.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(
        ["index", "path"] + [f"c{k}" for k in range(8)]
    )  # index, path, + labels (8 coords of the corners)
    csv_index = 0  # NOT updated when images fail to convert
    have_coords = self.labels.shape[1] == 8

    transaction = env.begin(write=True)
    try:
        for i, path in enumerate(
            tqdm(self.image_paths, position=0, desc="Saving precomputed examples")
        ):
            imdata = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
            if imdata is None:
                print(f"Couldn't read image at {path}.")
                continue

            resized = resize_img(imdata, self.target_h, self.target_w)
            if resized is None:
                print(f"Couldn't resize image at {path}.")
            # resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            bytes = pickle.dumps(resized)
            key = str(i).encode("ascii")
            transaction.put(key, bytes)

            exit(0)

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

    self.computed = True
    print("Precomputation complete.")