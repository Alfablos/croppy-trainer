#!/usr/bin env bash
# 
# 

outdir="./hires_compact"


precompute() {
  nix run . -- pc -o $outdir --height 1024 --width 768 --compute-corners --strict --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/smartdoc15/extended_smartdoc_dataset/train --purpose train --progress --compact --recess 0.01 --limit 1000 && nix run . -- pc -o $outdir --height 1024 --width 768 --compute-corners --strict --image-extension '_in.png' --label-extension '_gt.png' --architecture resnet --data-root ~/Downloads/smartdoc15/extended_smartdoc_dataset/validation --purpose val --progress --compact --recess 0.01 --limit 1000
}

train() {
  nix run . -- train --out-dir ./hires_out --db ./hires_compact/training_data/data_resnet_training_1000x1024x768_compacted.lmdb --valdb ./hires_compact/validation_data/data_resnet_validation_1000x1024x768_compacted.lmdb -a resnet --lr 0.0001 --dropout 0.1 -e 10 --tensorboard --progress --device gpu -H --debug 3
}

full() {
  precompute
  train
}

case "$1" in
  "pc" | "precompute" | "preprocess")
    echo "Running \`Precompute\`."
    precompute
    ;;
  "train" | "training")
    train
    ;;
  *)
    echo "Running \`Full\`."
    full
    ;;
esac