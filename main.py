from jinja2.nodes import FromImport
import argparse
from pathlib import Path

# from data import SmartDocDatasetResnet
from utils import Precision
from crawler import crawl
from preprocessor import precompute
from architecture import Architecture


# complete this function
def run_crawl(args):
    crawl(
        root=Path(args.data_root),
        images_ext=args.image_extension,
        labels_ext=args.label_extension,
        output=args.output,
        precision=Precision.from_str(args.precision),
        compute_corners=args.compute_corners,
        check_normalization=args.check_normalization,
        verbose=args.verbose,
    )

def run_precompute(args):
    if not args.data_map:
        crawler_config = {
            "root": Path(args.data_root),
            "images_ext": args.image_extension,
            "labels_ext": args.label_extension,
            "precision": Precision.from_str(args.precision),
            "compute_corners": args.compute_corners,
            "check_normalization": args.check_normalization,
            "verbose": args.verbose
        }
    else:
        crawler_config = None

    kwargs = {}
    if args.commit_frequency:
        kwargs["commit_freq"] = int(args.commit_frequency)
    if args.workers:
        kwargs["n_workers"] = int(args.workers)

    precompute(
        architecture=Architecture.from_str(args.architecture),
        db_output_dir=args.output_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        dataset_map_csv=args.data_map,
        crawler_config=crawler_config,
        dry_run=args.dry_run,
        verbose=args.verbose,
        compute_corners=args.compute_corners,
        strict=args.strict,
        precision=Precision.from_str(args.precision)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="croppy-trainer")
    # build-ds --data-root /home/antonio/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --iext '_in.png' --lext='_gt.png' -o ./dataset.csv -v -n -c
    supbparsers = parser.add_subparsers(
        title="supbcommands", help="Croppy training utilities"
    )

    ## crawl ##
    crawl_cmd = supbparsers.add_parser(
        name="crawl", aliases=["build-ds"], help="Builds a dataset CSV file"
    )
    precompute_cmd = supbparsers.add_parser(
        "precompute", aliases=["pc"], help="Prepare dataset for training"
    )

    preprocess_or_both = precompute_cmd.add_mutually_exclusive_group(required=True)
    preprocess_or_both.add_argument(
        "--data-map", "-m", help="Path to an existing index CSV (skips crawling)"
    )
    preprocess_or_both.add_argument(
        "--data-root", "-r", help="Root directory to scan for images (triggers crawler)"
    )

    crawl_cmd.add_argument("--data-root", "--root", "-r", required=True)
    crawl_cmd.add_argument(
        "--image-extension",
        "--image-ext",
        "--iext",
        "-i",
        required=True,
        help="The final part common to all image file names. E.g. '_train.png'",
    )
    crawl_cmd.add_argument(
        "--label-extension",
        "--label-ext",
        "--lext",
        "-l",
        required=True,
        help="The final part common to all label file names. E.g. '_train_label.png'",
    )
    crawl_cmd.add_argument(
        "-o", "--output", required=True, help="Where to save the CSV file"
    )
    crawl_cmd.add_argument("--precision", "-p", required=False, default="f32")
    crawl_cmd.add_argument("--compute-corners", "-c", action="store_true")
    crawl_cmd.add_argument("--check-normalization", "-n", action="store_true")
    crawl_cmd.add_argument("--verbose", "-v", action="store_true")
    crawl_cmd.set_defaults(func=run_crawl)

    ## precompute (crawler options) ## # the options of the crawler are only read if --csv is not set
    precompute_cmd.add_argument("--precision", "-p", required=False, default="f32")
    precompute_cmd.add_argument("--compute-corners", "-c", action="store_true")
    precompute_cmd.add_argument("--check-normalization", "-n", action="store_true")
    precompute_cmd.add_argument(
        "--image-extension",
        "--image-ext",
        "--iext",
        "-i",
        required=True,
        help="The final part common to all image file names. E.g. '_train.png'",
    )
    precompute_cmd.add_argument(
        "--label-extension",
        "--label-ext",
        "--lext",
        "-l",
        required=True,
        help="The final part common to all label file names. E.g. '_train_label.png'",
    )

    # true precompute_cmd arguments
    precompute_cmd.add_argument(
        "-o", "--output-dir", required=True, help="Where to save LMDB and CSV files"
    )
    precompute_cmd.add_argument("--architecture", "--arch", "-a", required=True)
    precompute_cmd.add_argument("--target-height", "--height", type=int, required=True)
    precompute_cmd.add_argument("--target-width", "--width", type=int, required=True)
    precompute_cmd.add_argument("--commit-frequency", "--commit-freq", required=False)
    precompute_cmd.add_argument("--dry-run", required=False, action="store_true")
    precompute_cmd.add_argument("--verbose", "-v", required=False, action="store_true")
    precompute_cmd.add_argument("--strict", "-s", required=False, action="store_true")
    precompute_cmd.add_argument("--workers", "--threads", "--n-workers", "--n-threads", "-w", required=False)
    precompute_cmd.set_defaults(func=run_precompute)

    args = parser.parse_args()
    # print(args)
    args.func(args)
