{ pkgs, ... }:
let
  lib = pkgs.lib;

  # General variables
  runCmd = "python croppy.py";
  architecture = "resnet";

  verbose = false;
  progress = true;
  cpuCount = 16;

  # Precompute variables
  precomputeOutputDir = "./croppy_" + (if compact then "compact_" else "") + h + "x" + w;
  h = "512"; # "1024";
  w = "512"; # "768";
  strict = false;
  compact =
    if
      (lib.strings.toLower store) == "lmdb" # while a simple copy is implemented to mimic compacting a lmdb store for an aroow store,
    # --compact doesn't make sense for arrow stores
    then
      true
    else
      false;
  commit_frequency = "100";
  precomputeWorkers = toString cpuCount;
  limit = "0";
  store = "arrow"; # also the file extension: .lmdb .arrow

  # Training variables
  training_limit = "0";
  trainingOutputDir =
    "croppy_"
    + h
    + "x"
    + w
    + "_learningRate-"
    + learning_rate
    + "_dropout-"
    + dropout
    + "_loss-"
    + loss_function
    + "_epochs-"
    + epochs
    + (if hard_validation then "_hardvalidation" else "")
    + "_${device}";
  storePath =
    purpose:
    precomputeOutputDir
    + "/${purpose}_data/data_${architecture}_${purpose}_"
    + "${h}x${w}"
    + (if compact then "_compacted" else "")
    + "."
    + lib.strings.toLower store;
  loss_function = "invariant_smooth_mae";
  learning_rate = "0.0001";
  dropout = "0.25";
  epochs = "100";
  workers = toString (cpuCount / 2);
  batch_size = "192";
  device = "gpu";
  debug = "3";
  checkpoint = "3";
  tensorboard = true;
  hard_validation = true;

  precomputeCmd = ''
    ${runCmd} precompute \
      -o ${precomputeOutputDir} \
      --height ${h} \
      --width ${w} \
      ${if strict then "--strict" else "--no-strict"} \
      --architecture ${architecture} \
      ${if verbose then "--verbose" else ""} \
      ${if progress then "--progress" else ""} \
      ${if compact then "--compact" else ""} \
      --commit-frequency ${commit_frequency} \
      --workers ${precomputeWorkers} \
      ${if limit != "0" then "--limit ${limit}" else ""}
  '';

  config = {
    precompute = {
      inherit
        strict
        architecture
        verbose
        progress
        compact
        commit_frequency
        ;
      output_dir = precomputeOutputDir;
      target_h = h;
      target_w = w;
      workers = precomputeWorkers;
    };
    training = {
      inherit
        architecture
        loss_function
        learning_rate
        batch_size
        dropout
        epochs
        workers
        verbose
        progress
        tensorboard
        hard_validation
        debug
        checkpoint
        ;
      output_dir = trainingOutputDir;
      training_store = storePath "training";
      validation_store = storePath "validation";
    };
  };
in
# with builtins;
# toFile "config.json" (toJSON config)

pkgs.writeScript "quick-run" ''
  #!/usr/bin/env bash

  precompute() {
    ${precomputeCmd}
  }

  train() {
    ${runCmd} train \
      --out-dir ${trainingOutputDir}  \
      --db ${storePath "training"} \
      --valdb ${storePath "validation"} \
      --architecture ${architecture} \
      --loss-function ${loss_function} \
      --learning-rate ${learning_rate} \
      --dropout ${dropout} \
      --epochs ${epochs} \
      --workers ${workers} \
      --batch-size ${batch_size} \
      ${if tensorboard then "--tensorboard" else ""} \
      ${if verbose then "--verbose" else ""} \
      ${if progress then "--progress" else ""} \
      --device ${device} \
      ${if hard_validation then "--hard-validation" else ""} \
      ${if training_limit != "0" then "--limit ${training_limit}" else ""} \
      ${if debug != "0" then "--debug ${debug}" else ""} \
      ${if checkpoint != "0" then "--checkpoint ${checkpoint}" else ""} 
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

      if [[ -f "${storePath "training"}" ]] && [[ -f "${storePath "validation"}" ]]; then
        echo "Stores found. Skipping precompute."
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
