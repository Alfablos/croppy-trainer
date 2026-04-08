import math

from functools import partial
from os import cpu_count

from concurrent.futures.process import ProcessPoolExecutor
from typing import Any, Iterable

import os
from pathlib import Path

import pandas as pd
import cv2
from tqdm import tqdm


from architecture import Architecture
from common import DataSource
import utils
from utils import Device


def chunks_from_list(items: Iterable[Any], n_chunks: int) -> list[Any]:
    if n_chunks <= 0:
        return []
    items = list(items)
    n = len(items)
    # print(f"chunks_from_list: received {n} paths to split in {n_chunks} chunks.")

    if n < n_chunks:
        return [p for p in items]

    chunk_size, remainder = (len(items) // n_chunks, len(items) % n_chunks)

    result = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        result.append(items[start:end])
    if remainder > 0:
        result.append(items[n_chunks * chunk_size :])

    return result


def process_chunk(
    rows: list[dict],
    label_type: str,
    verbose: bool,
):
    if verbose:
        print(f"Runner: got {len(rows)} objects.")
    computed_rows = []

    for row in rows:
        try:
            # Check if coordinates are already present
            has_coords = (
                row.get("x1") is not None
                and row.get("y1") is not None
                and row.get("x2") is not None
                and row.get("y2") is not None
                and row.get("x3") is not None
                and row.get("y3") is not None
                and row.get("x4") is not None
                and row.get("y4") is not None
            )

            if label_type == "coordinates":
                if has_coords:
                    # Already have coordinates, just pass through
                    pass
                else:
                    if row.get("label_path") is None:
                        # Cannot compute coords without mask and don't have coords
                        raise ValueError(
                            f"Cannot compute corner coordinates for {row.get('image_path')}: no label_path and no pre-computed coordinates."
                        )

                    mask = cv2.imread(
                        filename=str(row["label_path"]), flags=cv2.IMREAD_GRAYSCALE
                    )

                    coords = utils.coords_from_segmentation_mask(
                        mask,
                        device=Device.CPU,
                    )
                    fields = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
                    for coord_name, value in zip(fields, coords):
                        row[coord_name] = value
            else:
                # TODO: Implement mask generation from coordinates if needed
                raise NotImplementedError("Generating masks is not supported yet")

            computed_rows.append(row)
        except KeyboardInterrupt:
            if computed_rows:
                print("Execution stopped by user")
                exit(0)
        except Exception as e:
            print(f"Error processing row {row.get('image_path')}: {e}")
            continue

    if verbose:
        print(f"Runner: finished computing {len(computed_rows)} objects.")
    return computed_rows


def _crawl_worker(payload):
    f, args = payload
    return f(args)


def crawl(
    data_sources: list[DataSource],
    architecture: Architecture,
    output: str | None = None,
    limit: int | None = None,
    chunks: int = math.floor(cpu_count() / 2),
    check_normalization=True,
    verbose=False,
    progress=False,
):
    """
    Crawls multiple data sources to find and pair image files with their corresponding labels (masks or coordinates),
    optionally computes normalized document corner coordinates from segmentation masks, and saves the results to a CSV file.

    Args:
        data_sources: List of DataSource objects to crawl
        architecture: The model architecture
        output: Output CSV path (for internal use only; normally derived from data_sources)
        limit: Limit number of items
        chunks: Number of parallel chunks
        check_normalization: Whether to check normalization
        verbose: Verbose output
        progress: Show progress bar
    """
    # Determine output path from data source if not explicitly provided
    if output is None:
        if len(data_sources) == 1:
            output = data_sources[0].crawl_output_path
        else:
            raise ValueError(
                "When crawling multiple sources, output path must be provided explicitly. "
                "Consider crawling sources individually to use their configured paths."
            )

    if os.path.exists(output):
        print(f"Output file {output} exists. Refusing to continue.")
        exit(2)

    all_rows = []

    label_type = architecture.label_type

    for ds in data_sources:
        if verbose:
            print(f"Fetching from source: {ds.name}")

        ds_err = ds.check()
        if ds_err:
            raise RuntimeError(ds_err)

        ds_rows = ds.fetch(architecture)

        # Apply source-level exclude_patterns (regexes matched via re.search
        # against DataRow.image_path). Always-on log because dropping data is
        # a material configuration the user should see confirmed.
        if getattr(ds, "exclude_patterns", None):
            before = len(ds_rows)
            ds_rows = [r for r in ds_rows if not ds.should_exclude(r.image_path)]
            excluded = before - len(ds_rows)
            print(
                f"  [{ds.name}] excluded {excluded}/{before} rows "
                f"matching {ds.exclude_patterns}"
            )

        # Convert DataRow objects to dicts for processing/serialization
        all_rows.extend([r.to_dict() for r in ds_rows])

    # not very optimized for immense datasets!
    if limit:
        all_rows = all_rows[:limit]

    if verbose:
        print(f"Found {len(all_rows)} total examples.")

    if progress:
        progress_bar = tqdm(total=len(all_rows), desc="Processing examples")

    output_p = Path(output)
    if not output_p.parent.exists():
        output_p.parent.mkdir(parents=True)

    ## Parallelization
    # Split list of dicts into chunks
    image_label_pairs = chunks_from_list(all_rows, chunks)
    if verbose:
        print(f"Split dataset into {len(image_label_pairs)} batches.")

    task = partial(
        process_chunk,
        label_type=label_type,
        verbose=verbose,
    )

    payload = [(task, pair) for pair in image_label_pairs]
    to_write = []

    with ProcessPoolExecutor(max_workers=chunks) as executor:
        compute_tasks = executor.map(_crawl_worker, payload)
        if verbose:
            print(f"Starting {chunks} workers...")

        try:
            returned = 0
            # compute
            for result in compute_tasks:  # result is a chunk of processed rows
                returned += 1
                if verbose:
                    print(
                        f"Worker {returned} returned results of length {len(result)}."
                    )
                to_write.extend(result)
                if progress:
                    progress_bar.update(len(result))

        except KeyboardInterrupt:
            print("User interrupted crawling.")
            executor.shutdown(wait=False, cancel_futures=True)
        finally:
            if progress:
                progress_bar.close()

    if verbose:
        print("Done crawling. Saving.")
    save_to_csv(to_write, output, "w")


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
