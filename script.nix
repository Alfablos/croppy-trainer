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
  runCmd = "python croppy.py";
  architecture = "resnet";
  datasetLengths = {
    training = "22092";
    validation = "7430";
  };
  precomputeDataRoot = "~/Downloads/smartdoc15/extended_smartdoc_dataset";
  verbose = true;
  progress = false;
  cpuCount = 16;

  # Precompute variables
  precomputeOutputDir =
    "./croppy_"
    + (if compact then "compact_" else "")
    + (if limit != "0" then limit else datasetLengths.training)
    + "x"
    + h
    + "x"
    + w
    + "_recess"
    + recess;
  h = "512"; # "1024";
  w = "512"; # "768";
  iext = "_in.png";
  lext = "_gt.png";
  recess = "0";
  compute_corners = true;
  strict = true;
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
  training_limit = "100";
  trainingOutputDir =
    "croppy_"
    + (if training_limit == "0" then datasetLengths.training else training_limit)
    + "x"
    + h
    + "x"
    + w
    + "_recess-"
    + recess
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
    + (if limit != "0" then limit else datasetLengths.${purpose})
    + "x${h}x${w}"
    + (if compact then "_compacted" else "")
    + "."
    + lib.strings.toLower store;
  loss_function = "invariant_smooth_mae";
  learning_rate = "0.0001";
  dropout = "0.25";
  epochs = "100";
  workers = toString (cpuCount / 2);
  batch_size = "64";
  device = "gpu";
  debug = "5";
  checkpoint = "5";
  tensorboard = true;
  hard_validation = true;

  precomputeLoop = map (
    purpose:
    let
      dataRootSubdir = datasetSubdir purpose;
    in
    # ${if limit != "0" then "--limit ${limit} \\" else "\\"}
    ''
      ${runCmd} precompute \
        -o ${precomputeOutputDir} \
        --height ${h} \
        --width ${w} \
        ${if compute_corners then "--compute-corners" else ""} \
        ${if strict then "--strict" else ""} \
        --image-extension '${iext}' \
        --label-extension '${lext}' \
        --architecture ${architecture} \
        --data-root ${precomputeDataRoot + "/" + dataRootSubdir} \
        --purpose ${purpose} \
        ${if verbose then "--verbose" else ""} \
        ${if progress then "--progress" else ""} \
        ${if compact then "--compact" else ""} \
        --commit-frequency ${commit_frequency} \
        --workers ${precomputeWorkers} \
        --recess ${recess} ${if limit != "0" then "--limit ${limit} " else " "}'' # no newline here!
  ) purposes;

  precomputeForPurpose = purpose:
    {
      inherit
        compute_corners
        strict
        architecture
        purpose
        verbose
        progress
        compact
        commit_frequency
        recess
        ;
      data_root = precomputeDataRoot + "/" + datasetSubdir purpose;
      output_dir = precomputeOutputDir;
      target_h = h;
      target_w = w;
      image_extension = iext;
      label_extension = lext;
      workers = precomputeWorkers;
    };
  
  config =  {
    precomputations = map (p: precomputeForPurpose p) purposes;
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
with builtins;
toFile "config.json" (toJSON config)


# pkgs.writeScript "quick-run" ''
#   #!/usr/bin/env bash
#
#   precompute() {
#     ${lib.strings.concatStringsSep "&& \\\n  " precomputeLoop}
#   }
#
#   train() {
#     ${runCmd} train \
#       --out-dir ${trainingOutputDir}  \
#       --db ${storePath "training"} \
#       --valdb ${storePath "validation"} \
#       --architecture ${architecture} \
#       --loss-function ${loss_function} \
#       --learning-rate ${learning_rate} \
#       --dropout ${dropout} \
#       --epochs ${epochs} \
#       --workers ${workers} \
#       --batch-size ${batch_size} \
#       ${if tensorboard then "--tensorboard" else ""} \
#       ${if verbose then "--verbose" else ""} \
#       ${if progress then "--progress" else ""} \
#       --device ${device} \
#       ${if hard_validation then "--hard-validation" else ""} \
#       ${if trainLimit != 0 then "--limit ${training_limit}" else ""} \
#       ${if debug != 0 then "--debug ${debug}" else ""} \
#       ${if checkpoint != 0 then "--checkpoint ${checkpoint}" else ""} 
#   }
#
#
#   full() {
#     precompute
#     train
#   }
#
#   training_output_dir="${trainingOutputDir}"
#   precompute_output_dir="${precomputeOutputDir}"
#
#   case "$1" in
#     "pc" | "precompute" | "preprocess")
#       echo "Running \`Precompute\`."
#       precompute
#       ;;
#     "train" | "training")
#       train
#       ;;
#     *)
#       if [[ -d $training_output_dir ]] && [[ `ls $training_output_dir` != "" ]]; then
#         echo "Training output dir exists and is not empty. Refusing to continue."
#         exit 2
#       fi
#       
#       if [[ -d $precompute_output_dir ]]; then
#         train
#       else
#         echo "Running FULL"
#         echo "Running precompute before training."
#         precompute
#         train
#       fi
#       ;;
#   esac
# ''
