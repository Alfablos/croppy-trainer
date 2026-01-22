from multiprocessing import cpu_count
from cli import dependencies, run_crawl, run_precompute, run_train
import subprocess
from sympy.printing.pretty.pretty_symbology import sup
from jinja2.nodes import FromImport
import argparse
from pathlib import Path

# from data import SmartDocDatasetResnet
from common import Precision
from crawler import crawl
from preprocessor import precompute
from architecture import Architecture


if __name__ == "__main__":
    # python main.py pc -o ./ --height 512 --width 384 --compute-corners --strict --precision f32 --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --purpose train -v
    # python main.py pc -o ./ --height 512 --width 384 --compute-corners --strict --precision f32 --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/validation --purpose val -v
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda _: parser.print_help())
    # build-ds --data-root /home/antonio/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --iext '_in.png' --lext='_gt.png' -o ./dataset.csv -v -n -c
    supbparsers = parser.add_subparsers(
        title="supbcommands", help="Croppy training utilities"
    )
    
    train_cmd = supbparsers.add_parser(
        name = "train", help="Trains Croppy"
    )
    
    dependencies_cmd = supbparsers.add_parser(
        name="dependencies", aliases = ["deps"], help="Get dependencies version"
    )
    dependencies_cmd.set_defaults(func=dependencies)


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
    precompute_cmd.add_argument("--precision", "--prec", required=True)
    precompute_cmd.add_argument("--target-height", "--height", type=int, required=True)
    precompute_cmd.add_argument("--target-width", "--width", type=int, required=True)
    precompute_cmd.add_argument("--commit-frequency", "--commit-freq", required=False, default=100)
    precompute_cmd.add_argument("--dry-run", required=False, action="store_true")
    precompute_cmd.add_argument("--verbose", "-v", required=False, action="store_true")
    precompute_cmd.add_argument("--strict", "-s", required=False, action="store_true")
    precompute_cmd.add_argument("--workers", "--threads", "--n-workers", "--n-threads", "-w", required=False, default=cpu_count())
    precompute_cmd.add_argument("--purpose", "-P", required=True, type=str)
    precompute_cmd.set_defaults(func=run_precompute)
    
    
    ## Train ##
    train_cmd.add_argument("--lmdb-path", "--db", required=True)
    train_cmd.add_argument("--validation-lmdb-path", "--valdb", required=False)
    train_cmd.add_argument("--architecture", "--arch", "-a", required=True)
    train_cmd.add_argument("--learning-rate", "--lrate", "--lr", required=True, type=float)
    train_cmd.add_argument("--epochs", "-e", required=True, type=int)
    train_cmd.add_argument("--output-file", "--output", "-o", required=False, help="Where to save the model weights")
    train_cmd.add_argument("--precision", "-p", required=False, default="f32")
    train_cmd.add_argument("--limit", required=False, type=int)
    train_cmd.add_argument("--workers", "-w", required=False, type=int, default=cpu_count())
    train_cmd.add_argument("--batch-size", "-b", required=False, type=int, default=64)
    train_cmd.add_argument("--device", "-d", required=False, type=str, default='cuda')
    train_cmd.add_argument("--dropout", required=False, type=float, default=0.3)
    train_cmd.add_argument("--verbose", "-v", action="store_true", required=False, default=False)
    train_cmd.add_argument("--progress", action="store_true", required=False, default=False)
    train_cmd.add_argument("--enable-tensorboard", "--with_tensorboard", "--tensorboard", "-B", action="store_true", required=False, default=False)
    train_cmd.add_argument("--tensorboard-logdir", "--logdir", required=False, type=str)
    train_cmd.set_defaults(func=run_train)
    

    args = parser.parse_args()
    args.func(args)
    
    
    
