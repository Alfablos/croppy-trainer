{ pkgs, ... }:
let
  lib = pkgs.lib;
  purposes = [
    "training"
    "validation"
  ];
  datasetSubdir =
    purpose:
    {
      training = "train";
      validation = "validation";
    }
    .${purpose};

  # General variables
  architecture = "resnet";
  datasetLengths = {
    training = "22092";
    validation = "7430";
  };
  precomputeDataRoot = "~/Downloads/smartdoc15/extended_smartdoc_dataset";
  verbose = false;
  progress = true;
  cpuCount = 16;

  # Precompute variables
  precomputeOutputDir =
    "./croppy_"
    + (if compact then "compact_" else "")
    + (if limit != "0" then limit else datasetLengths.training)
    + "_recess"
    + recess;
  h = "512";
  w = "384";
  iext = "_in.png";
  lext = "_gt.png";
  recess = "0.005";
  computeCorners = true;
  strict = true;
  compact = true;
  commitFrequency = "100";
  precomputeWorkers = toString cpuCount;
  limit = "0";

  # Train variables
  trainingOutputDir =
    "croppy_"
    + (if limit == "0" then datasetLengths.training else limit)
    + "x"
    + h
    + "x"
    + w
    + "_recess-"
    + recess
    + "_learningRate-"
    + learningRate
    + "_dropout-"
    + dropout
    + "_epochs-"
    + epochs
    + (if hardValidation then "_hardvalidation" else "")
    + "_${device}";
  storePath =
    purpose:
    precomputeOutputDir
    + "/${purpose}_data/data_${architecture}_${purpose}_"
    + (if limit != "0" then limit else datasetLengths.${purpose})
    + "x${h}x${w}"
    + (if compact then "_compacted" else "")
    + ".lmdb";
  loss_function = "mae";
  learningRate = "0.0001";
  dropout = "0.25";
  epochs = "30";
  workers = toString (cpuCount / 2);
  batchSize = "128";
  device = "gpu";
  debug = "2";
  tensorboard = true;
  hardValidation = true;

  precomputeLoop = map (
    purpose:
    let
      dataRootSubdir = datasetSubdir purpose;
    in
    # ${if limit != "0" then "--limit ${limit} \\" else "\\"}
    ''
      nix run . -- precompute \
        -o ${precomputeOutputDir} \
        --height ${h} \
        --width ${w} \
        ${if computeCorners then "--compute-corners" else ""} \
        ${if strict then "--strict" else ""} \
        --image-extension '${iext}' \
        --label-extension '${lext}' \
        --architecture ${architecture} \
        --data-root ${precomputeDataRoot + "/" + dataRootSubdir} \
        --purpose ${purpose} \
        ${if verbose then "--verbose" else ""} \
        ${if progress then "--progress" else ""} \
        ${if compact then "--compact" else ""} \
        --commit-frequency ${commitFrequency} \
        --workers ${precomputeWorkers} \
        --recess ${recess} ${if limit != "0" then "--limit ${limit}" else ""}'' # no newline here!
  ) purposes;

in
pkgs.writeScript "quick-run" ''
  #!/usr/bin/env bash

  precompute() {
    ${lib.strings.concatStringsSep "&& \\\n  " precomputeLoop}
  }

  train() {
    nix run . -- train \
      --out-dir ${trainingOutputDir}  \
      --db ${storePath "training"} \
      --valdb ${storePath "validation"} \
      --architecture ${architecture} \
      --loss-function ${loss_function} \
      --learning-rate ${learningRate} \
      --dropout ${dropout} \
      --epochs ${epochs} \
      --workers ${workers} \
      --batch-size ${batchSize} \
      ${if tensorboard then "--tensorboard" else ""} \
      ${if verbose then "--verbose" else ""} \
      ${if progress then "--progress" else ""} \
      --device ${device} \
      ${if hardValidation then "--hard-validation" else ""} \
      --debug ${debug}
  }


  full() {
    precompute
    train
  }

  training_output_dir="${trainingOutputDir}"
  precompute_output_dir="${precomputeOutputDir}"

  case "$1" in
    "pc" | "precompute" | "preprocess")
      echo "Running \`Precompute\`."
      precompute
      ;;
    "train" | "training")
      train
      ;;
    *)
      if [[ -d $training_output_dir ]] && [[ `ls $training_output_dir` != "" ]]; then
        echo "Training output dir exists and is not empty. Refusing to continue."
        exit 2
      fi
      
      if [[ -d $precompute_output_dir ]]; then
        train
      else
        echo "Running FULL"
        echo "Running precompute before training."
        precompute
        train
      fi
      ;;
  esac
''
