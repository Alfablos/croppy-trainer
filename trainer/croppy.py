import torch
from multiprocessing import cpu_count
from cli import run_crawl, run_precompute, run_train, run_predict, run_merge_stores
import argparse

# from data import SmartDocDatasetResnet


if __name__ == "__main__":
    # python main.py pc -o ./ --height 512 --width 384 --compute-corners --strict --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --purpose train -v
    # python main.py pc -o ./ --height 512 --width 384 --compute-corners --strict --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/validation --purpose val -v
    # python main.py train --out-dir ./ --db ./train_data/data_resnet_train.lmdb --valdb ./validation_data/data_resnet_validation.lmdb -a resnet --lr 0.001 -e 10 --tensorboard --progress --device gpu

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda _: parser.print_help())
    # build-ds --data-root /home/antonio/Downloads/extended_smartdoc_dataset/Extended\ Smartdoc\ dataset/train --iext '_in.png' --lext='_gt.png' -o ./dataset.csv -v -n -c
    supbparsers = parser.add_subparsers(
        title="supbcommands", help="Croppy training utilities"
    )

    train_cmd = supbparsers.add_parser(name="train", help="Trains Croppy")

    predict_cmd = supbparsers.add_parser(
        name="predict", help="Perform inference on the given image"
    )

    ## crawl ##
    crawl_cmd = supbparsers.add_parser(
        name="crawl", aliases=["build-ds"], help="Builds a dataset CSV file"
    )
    precompute_cmd = supbparsers.add_parser(
        "precompute", aliases=["pc"], help="Prepare dataset for training"
    )

    # Crawling is always done per-purpose from DATA_SOURCES in config.py

    crawl_cmd.add_argument(
        "--architecture", "--arch", "-a", required=True,
        help="Model architecture (resnet or unet), determines what label data to compute"
    )
    crawl_cmd.add_argument(
        "--purpose", "-p", required=True,
        choices=["training", "validation", "test"],
        help="Purpose to crawl (training/validation/test)"
    )
    crawl_cmd.add_argument(
        "--source",
        help="Specific data source name to crawl. If not specified, crawls all sources for the purpose."
    )
    crawl_cmd.add_argument("--check-normalization", "-n", action="store_true")
    crawl_cmd.add_argument(
        "--verbose", "-v", action="store_true", required=False, default=False
    )
    crawl_cmd.add_argument(
        "--progress", action="store_true", required=False, default=False
    )
    crawl_cmd.add_argument(
        "--limit", "-L", type=int, help="Limit the number of items to include"
    )
    crawl_cmd.set_defaults(func=run_crawl)

    ## precompute (crawler options) ## # the options of the crawler are only read if --csv is not set
    precompute_cmd.add_argument("--check-normalization", "-n", action="store_true")

    # true precompute_cmd arguments
    precompute_cmd.add_argument("--architecture", "--arch", "-a", required=True)
    precompute_cmd.add_argument("--target-height", "--height", type=int, required=True)
    precompute_cmd.add_argument("--target-width", "--width", type=int, required=True)
    precompute_cmd.add_argument("--purpose", "-p", required=True,
        choices=["training", "validation", "test"])
    precompute_cmd.add_argument(
        "--source",
        help="Specific data source name to precompute. If not specified, precomputes all sources for the purpose."
    )
    precompute_cmd.add_argument(
        "--commit-frequency", "--commit-freq", required=False, type=int, default=100
    )
    precompute_cmd.add_argument("--dry-run", required=False, action="store_true")
    precompute_cmd.add_argument(
        "--verbose", "-v", required=False, default=False, action="store_true"
    )
    precompute_cmd.add_argument(
        "--progress", required=False, default=False, action="store_true"
    )
    precompute_cmd.add_argument(
        "--strict",
        "-s",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Error if a single image fails to be processed. Use --no-strict to skip bad images.",
    )
    precompute_cmd.add_argument(
        "--workers",
        "--threads",
        "--n-workers",
        "--n-threads",
        "-w",
        required=False,
        type=int,
        default=cpu_count(),
    )

    precompute_cmd.add_argument(
        "--compact-store",
        "--compact-database",
        "--compact-db",
        "--compact",
        "-C",
        required=False,
        action="store_true",
        default=False,
        help="Avoids LMDB store sparsity to ensure compatibility with S3 storage and non sparse-tolerant storage backends."
        + "WARNING: Requires an amount of additional storage space equal to the size of the store actual (non sparse) content."
        + "Preprocessing duration might increase dramatically.",
    )
    precompute_cmd.add_argument(
        "--output-dir",
        "-o",
        required=False,
        default=None,
        help="Top-level output directory. Stores go under {dir}/training_data/ and {dir}/validation_data/. "
             "Defaults to config.PATHS if not specified.",
    )
    precompute_cmd.add_argument(
        "--limit", "-L", type=int, help="Limit the number of items to include"
    )
    precompute_cmd.set_defaults(func=run_precompute)

    ## Train ##
    train_cmd.add_argument(
        "--store-dir",
        "--dir",
        "-S",
        help="Directory containing precomputed per-source Arrow stores. "
             "If not specified, uses the default from config.PATHS['training']['precompute_output_dir'].",
    )
    train_cmd.add_argument("--architecture", "--arch", "-a", required=True)
    train_cmd.add_argument(
        "--learning-rate", "--lrate", "--lr", "-l", required=True, type=float
    )
    train_cmd.add_argument("--epochs", "-e", required=True, type=int)
    train_cmd.add_argument(
        "--output-directory",
        "--out-dir",
        "--output",
        "-o",
        required=True,
        help="Where to save the model weights and specs file",
    )
    train_cmd.add_argument(
        "--loss-function", "--loss-fn", "--loss", "-L", required=False, default="mse"
    )
    train_cmd.add_argument("--precision", "-p", required=False, default="f32")
    train_cmd.add_argument("--limit", required=False, type=int)
    train_cmd.add_argument(
        "--workers",
        "--threads",
        "--n-workers",
        "--n-threads",
        "-w",
        required=False,
        type=int,
        default=int(cpu_count() / 2),
        help="How many worker per dataset to instantiate",
    )
    train_cmd.add_argument("--batch-size", "-b", required=False, type=int, default=32)
    train_cmd.add_argument(
        "--device", "--dev", "-d", required=False, type=str, default="cuda"
    )
    train_cmd.add_argument(
        "--hard-validation",
        "--hard-val",
        "--hard",
        "-H",
        action="store_true",
        required=False,
        default=False,
        help="Perform the same transforms as the train set on the validation set, making it harder for the model to get a good score",
    )
    train_cmd.add_argument(
        "--verbose", "-v", action="store_true", required=False, default=False
    )
    train_cmd.add_argument("--debug", "-D", required=False, default=None)
    train_cmd.add_argument("--checkpoint", "-C", required=False, type=int, default=10)
    train_cmd.add_argument(
        "--progress", action="store_true", required=False, default=False
    )
    train_cmd.add_argument(
        "--enable-tensorboard",
        "--with_tensorboard",
        "--tensorboard",
        "-B",
        action="store_true",
        required=False,
        default=False,
    )
    train_cmd.add_argument(
        "--resume",
        "-R",
        required=False,
        default=None,
        help="Path to a .pth checkpoint file to resume training from.",
    )
    train_cmd.set_defaults(func=run_train)

    predict_cmd.add_argument("path")
    predict_cmd.add_argument(
        "--output", "-o", required=True, help="Where to save LMDB and CSV files"
    )
    predict_cmd.add_argument(
        "--checkpoint", "--config", "-c", required=True, help="Path to a .pth checkpoint file."
    )
    predict_cmd.add_argument(
        "--device",
        "--dev",
        "-d",
        required=False,
        help="Device to run inference on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    predict_cmd.set_defaults(func=run_predict)

    ## merge-stores ##
    merge_cmd = supbparsers.add_parser(
        "merge-stores",
        aliases=["merge"],
        help="Merge multiple Arrow store files into one",
    )
    merge_cmd.add_argument(
        "stores",
        nargs="+",
        help="Paths to .arrow store files to merge",
    )
    merge_cmd.add_argument(
        "--output", "-o", required=True, help="Output path for the merged .arrow file"
    )
    merge_cmd.set_defaults(func=run_merge_stores)

    args = parser.parse_args()
    args.func(args)
