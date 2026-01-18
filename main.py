import argparse
from pathlib import Path

# from data import SmartDocDatasetResnet
from utils import Precision
from crawler import crawl


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
    precompute_cmd = supbparsers.add_parser("precompute", aliases=["pc"], help="Prepare dataset for training")
    
    
    preprocess_or_both = precompute_cmd.add_mutually_exclusive_group(required=True)
    preprocess_or_both.add_argument("--csv", "-f", help="Path to an existing index CSV (skips crawling)")
    preprocess_or_both.add_argument("--data-root", "-r", help="Root directory to scan for images (triggers crawler)")
    
    
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
    crawl_cmd.add_argument("-o", "--output", required=True, help="Where to save the CSV file")
    crawl_cmd.add_argument("--precision", "-p", required=False, default="f32")
    crawl_cmd.add_argument("--compute-corners", "-c", action="store_true")
    crawl_cmd.add_argument("--check-normalization", "-n", action="store_true")
    crawl_cmd.add_argument("--verbose", "-v", action="store_true")
    crawl_cmd.set_defaults(func=run_crawl)

    ## precompute (crawler options) ## # the options of the crawler are only read if --csv is not set
    precompute_cmd.add_argument("--precision", "-p", required=False, default="f32")
    precompute_cmd.add_argument("--compute-corners", "-c", action="store_true")
    precompute_cmd.add_argument("--check-normalization", "-n", action="store_true")
    precompute_cmd.add_argument("--verbose", "-v", action="store_true")
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
    precompute_cmd.add_argument("-o", "--output-dir", required=True, help="Where to save LMDB and CSV files")


    args = parser.parse_args()
    # print(args)
    args.func(args)
