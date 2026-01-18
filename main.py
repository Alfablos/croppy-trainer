import argparse
from pathlib import Path

from data import build_csv, SmartDocDatasetResnet
from utils import Precision


# complete this function
def run_build_csv(args):
    build_csv(
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

    ## Build Dataset ##
    build_ds = supbparsers.add_parser(
        name="build-dataset", aliases=["build-ds"], help="Builds a dataset CSV file"
    )
    build_ds.add_argument("--data-root", "--root", "-r", required=True)
    build_ds.add_argument(
        "--image-extension",
        "--image-ext",
        "--iext",
        "-i",
        required=True,
        help="The final part common to all image file names. E.g. '_train.png'",
    )
    build_ds.add_argument(
        "--label-extension",
        "--label-ext",
        "--lext",
        "-l",
        required=True,
        help="The final part common to all label file names. E.g. '_train_label.png'",
    )
    build_ds.add_argument("--output", "-o", required=True)
    build_ds.add_argument("--precision", "-p", required=False, default="f32")
    build_ds.add_argument("--compute-corners", "-c", action="store_true")
    build_ds.add_argument("--check-normalization", "-n", action="store_true")
    build_ds.add_argument("--verbose", "-v", action="store_true")
    build_ds.set_defaults(func=run_build_csv)

    ## Precompute Examples

    args = parser.parse_args()
    args.func(args)
