import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import tv_tensors

from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from pathlib import Path
import torch.distributed.optim.post_localSGD_optimizer
import tqdm
from loss import loss_from_str
from architecture import Architecture
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
import torchvision.models as visionmodels
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from common import Device
from loss import PermutationInvariantLoss


class CroppyNet(
    nn.Module
):

    def __init__(
        self,
        architecture: Architecture,
        loss_fn: Callable,
        target_device: Device,
        images_height: int,
        images_width: int,
        learning_rate: float,
    ):
        super().__init__()

        self.architecture = architecture
        self.loss_fn = loss_fn
        self.target_device = target_device
        self.images_height: int = images_height
        self.images_width = images_width
        self.learning_rate = learning_rate
        self.weights = config.backbone_weights

        # Backbone — truncated at layer2 for higher spatial resolution (64×64 at 512 input)
        base = config.backbone_model_fn(weights=config.backbone_weights, progress=True)
        backbone_layers = list(base.children())[:6]  # conv1, bn1, relu, maxpool, layer1, layer2
        self.model = nn.Sequential(*backbone_layers)

        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Selectively unfreeze layer2 (index 5) to learn corner-relevant features
            if not config.freeze_backbone_layer2:
                for param in self.model[5].parameters():
                    param.requires_grad = True

        ds = config.backbone_downsample_factor
        if (images_height % ds != 0) or (images_width % ds != 0):
            if architecture == Architecture.RESNET:
                raise ValueError(
                    f"Resnet requires images height and width to be divisible by {ds}! Current values: h = {images_height}, w = {images_width}"
                )

        if config.coord_conv:
            self.add_coords = config.AddCoordChannels()

        self.fc = config.head
        self.soft_argmax = config.SoftArgmax2D()

    def forward(self, x):
        x = self.model(x)                          # backbone: (B, 128, 64, 64)
        if config.coord_conv:
            x = self.add_coords(x)                  # coord grids: (B, 130, 64, 64)
        heatmaps = self.fc(x)                       # conv head: (B, 4, 128, 128)
        coords = self.soft_argmax(heatmaps)          # (B, 4, 2)
        return coords.flatten(start_dim=1)           # (B, 8)

    def loss_function(self):
        if isinstance(self.loss_fn, L1Loss):
            return "L1Loss"
        elif isinstance(self.loss_fn, MSELoss):
            return "MSELoss"
        elif isinstance(self.loss_fn, PermutationInvariantLoss):
            return "Invariant" + self.loss_fn.inner_to_str()
        else:
            raise UnimplementedError

    @staticmethod
    def from_trained_config(checkpoint: dict, device: Device):
        model: CroppyNet = CroppyNet(
            loss_fn=loss_from_str(checkpoint["loss_fn"]),
            architecture=Architecture.from_str(checkpoint["architecture"]),
            target_device=device,
            images_height=checkpoint["images_height"],
            images_width=checkpoint["images_width"],
            learning_rate=checkpoint["current_learning_rate"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device.value)  # adds a validation step


def save_checkpoint(
    epoch_progress: tuple[int, int],
    epoch_losses: tuple[
        float, float | None
    ],  # TODO: check the output type of loss functions
    run_name: str,
    out_dir: str,
    model: CroppyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None = None,
):
    epoch, epochs = epoch_progress
    epoch_train_loss, epoch_val_loss = epoch_losses

    checkpoint_name = f"{run_name}_epoch_{epoch + 1}_of_{epochs}"
    checkpoint_file = str(out_dir) + "/" + checkpoint_name + ".pth"

    checkpoint = {
        "architecture": f"{model.architecture}",
        "images_height": model.images_height,
        "images_width": model.images_width,
        "total_epochs": epochs,
        "epoch": epoch + 1,
        "loss_fn": model.loss_function(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": epoch_train_loss,
        "val_loss": epoch_val_loss,
        "initial_learning_rate": model.learning_rate,
        "current_learning_rate": optimizer.param_groups[0]["lr"],
    }

    torch.save(checkpoint, checkpoint_file)


@torch.no_grad()
def validation_data(
    model,
    loader,
    loss_fn,
    device: Device,
    verbose: bool,
    gpu_transforms,
    debug: bool,
    visual_debug_path: str,
    s_writer: SummaryWriter | None = None,
    epoch: int = 0,
    val_source_boundaries: list[tuple[str, int]] | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    batch_n = 0
    samples_seen = 0

    # Per-source loss tracking (ConcatDataset is sequential, shuffle=False)
    source_losses: dict[str, float] = {}
    source_batch_counts: dict[str, int] = {}
    if val_source_boundaries:
        for name, _ in val_source_boundaries:
            source_losses[name] = 0.0
            source_batch_counts[name] = 0

    # Debug: sample from evenly-spaced batches for representative dataset coverage
    # (ConcatDataset is sequential, so last-batch-only would only show the last dataset)
    debug_images = {}
    if debug:
        n_sample_points = min(3, len(loader))
        debug_batch_indices = {
            round(len(loader) * (i + 1) / n_sample_points)
            for i in range(n_sample_points)
        }

    for images, labels in loader:
        batch_n += 1
        batch_size = len(images)
        if verbose:
            print(f"Training (validation): starting batch {batch_n} of {len(loader)}")
        images, labels = images.to(device.value), labels.to(device.value)
        h, w = images.shape[-2:]
        labels_wrapped = tv_tensors.KeyPoints(
            labels.to("cuda"), canvas_size=(h, w), dtype=torch.float32
        )

        images, labels = gpu_transforms(images.to("cuda"), labels_wrapped)
        new_h, new_w = images.shape[-2:]
        labels = labels.as_subclass(torch.Tensor)
        labels = (labels / torch.tensor([new_w, new_h], device="cuda")).flatten(
            start_dim=1
        )

        preds = model(images)
        batch_loss = loss_fn(preds, labels).item()
        val_loss += batch_loss

        # Attribute batch loss to the source containing the batch midpoint
        if val_source_boundaries:
            batch_midpoint = samples_seen + batch_size // 2
            for name, end_idx in val_source_boundaries:
                if batch_midpoint < end_idx:
                    source_losses[name] += batch_loss
                    source_batch_counts[name] += 1
                    break
        samples_seen += batch_size

        # Collect debug samples from evenly-spaced batches
        if debug and batch_n in debug_batch_indices:
            end = min(4, len(images))
            batch_dict = utils.dump_training_batch(
                images=images[0:end], labels=labels[0:end], preds=preds[0:end],
                epoch=epoch, batch_idx=batch_n,
            )
            debug_images.update(batch_dict)

    # Write all collected debug images after the loop
    if debug_images:
        for fname, data in debug_images.items():
            cv2.imwrite(visual_debug_path + f"/validation_{fname}", data)
            if s_writer is not None:
                rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                s_writer.add_image(
                    f"validation/{fname}", rgb, global_step=epoch + 1, dataformats="HWC"
                )

    # Compute per-source average losses
    per_source_losses = {}
    for name in source_losses:
        if source_batch_counts[name] > 0:
            per_source_losses[name] = source_losses[name] / source_batch_counts[name]

    return val_loss / len(loader), per_source_losses


def train(
    model: CroppyNet,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    train_gpu_transforms,
    val_gpu_transforms,
    epochs: int,
    out_dir: str,
    train_len: int,  # only to append the information to filename and specs
    hard_validation: bool,
    debug: int | None,
    checkpoint: int | None,
    resume_from: dict | None = None,
    with_tensorboard: bool = False,
    verbose=False,
    progress=False,
    val_source_boundaries: list[tuple[str, int]] | None = None,
):
    out_dir = out_dir.rstrip("/")
    if with_tensorboard:
        tensorboard_logdir = out_dir + "/runs"

    if verbose:
        print("Starting training with parameters:")
        for k, v in locals().items():
            print(f"==> {k}: {v}")
            print()

    # Avoids division by 0 later on if debug or checkpoint are 0
    if debug == 0:
        debug = None
    if checkpoint == 0:
        checkpoint = None

    start_epoch = 0
    if resume_from:
        start_epoch = resume_from["epoch"]  # 1-indexed in checkpoint, used as 0-indexed start
        print(f"Resuming from epoch {start_epoch}")

    run_name = f"{model.architecture}_{model.loss_function()}_{model.learning_rate}lr_{epochs}epochs_{train_len}x{model.images_height}x{model.images_width}"
    print(f"Starting run {run_name}")
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    if not resume_from:
        out_dir_files = next(out_dir.walk())[2]
        for f in out_dir_files:
            if f.startswith(run_name) and f.endswith(".pth"):
                raise FileExistsError(
                    f"Refusing to overwrite files of a previous run. {f} already exists."
                )

    if with_tensorboard:
        s_writer = SummaryWriter(log_dir=tensorboard_logdir)
        per_source_tags = [f"loss/validation/{name}" for name, _ in (val_source_boundaries or [])]
        s_writer.add_custom_scalars({
            "Loss": {
                "train_vs_val": ["Multiline", ["loss/train", "loss/validation"]],
                "val_per_source": ["Multiline", ["loss/validation"] + per_source_tags],
            },
            "Diagnostics": {
                "pred_range": ["Multiline", ["diagnostics/pred_min", "diagnostics/pred_max"]],
            },
        })
        url = utils.launch_tensorboard(tensorboard_logdir)
        if verbose:
            print(f"Tensorboard is listening at {url}")

    if progress:
        epochs_iter = tqdm.trange(start_epoch, epochs, position=0, initial=start_epoch, total=epochs)
    else:
        epochs_iter = range(start_epoch, epochs)

    visual_debug_training_subdir = "/visual_debug_training"
    visual_debug_validation_subdir = "/visual_debug_validation"
    visual_debug_training_path = f"{out_dir}" + visual_debug_training_subdir
    visual_debug_validation_path = f"{out_dir}" + visual_debug_validation_subdir
    Path(visual_debug_training_path).mkdir(parents=True, exist_ok=True)
    Path(visual_debug_validation_path).mkdir(parents=True, exist_ok=True)

    # THE MODEL MUST BE MOVED TO THE RIGHT DEVICE BEFORE INITIALIZING THE OPTIMIZER
    model = model.to(model.target_device.value)
    optimizer = Adam(model.parameters(), lr=model.learning_rate, weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode=config.scheduler_mode, factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=config.scheduler_threshold)

    if resume_from:
        optimizer.load_state_dict(resume_from["optimizer_state_dict"])
        if resume_from.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_from["scheduler_state_dict"])

    for epoch in epochs_iter:
        model.train()

        if verbose:
            print(f"Starting epoch {epoch + 1}.")

        cumulative_train_loss = 0.0

        debug_fn = (
            lambda i, l, p: utils.dump_training_batch(
                images=i,
                labels=l,
                preds=p,
                epoch=epoch,
                batch_idx=batch_n,
                # purpose=purpose,
                # output_dir=f"{out_dir}/visual_debug_{purpose}",
            )
            if debug
            else None
        )

        if progress:
            sub_bar = tqdm.tqdm(total=len(train_dataloader), leave=True, position=1)
        batch_n = 0
        try:
            for images, labels in train_dataloader:
                batch_n += 1
                if verbose:
                    print(
                        f"Training: starting batch {batch_n} of {len(train_dataloader)}"
                    )
                images, labels = (
                    images.to(model.target_device.value),
                    labels.to(model.target_device.value),
                )
                h, w = images.shape[-2:]

                # For some reason labels are reconverted to normal tensors
                # they need to be KeyPoints or transforms will ignore them
                labels_wrapped = tv_tensors.KeyPoints(
                    labels.to("cuda"), canvas_size=(h, w), dtype=torch.float32
                )
                # the gpu has to handle transforms
                with torch.no_grad():
                    images, labels = train_gpu_transforms(images.to("cuda"), labels_wrapped)
                new_h, new_w = images.shape[-2:]
                # See https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_tv_tensors.html#but-i-want-a-tvtensor-back
                # normalization may be ineffective on Keypoints, need to unwrap the underlying tensor
                labels = labels.as_subclass(torch.Tensor)
                labels = (labels / torch.tensor([new_w, new_h], device="cuda")).flatten(
                    start_dim=1
                )
                # No clamping, situations like x > w will be handled post-prediction
                # labels = torch.clamp(labels.flatten(start_dim=1), 0.0, 1.0)

                optimizer.zero_grad()
                preds = model(images)
                loss = model.loss_fn(preds, labels)
                loss.backward()
                if config.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
                optimizer.step()
                cumulative_train_loss += loss.item()

                if progress:
                    sub_bar.update(1)

                # debug dump only on LAST minibatch of each epoch if epoch % debug == 0
                if (
                    debug
                    and (epoch + 1) % debug == 0
                    and batch_n == len(train_dataloader)
                ):
                    end = min(10, len(images))
                    # debug_fn(i=images[0:end], l=labels[0:end], p=preds[0:end], purpose=Purpose.TRAINING)
                    img_dict = debug_fn(
                        i=images[0:end], l=labels[0:end], p=preds[0:end]
                    )
                    for fname, data in img_dict.items():
                        cv2.imwrite(
                            f"{visual_debug_training_path}/training_{fname}", data
                        )
                        if with_tensorboard:
                            rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                            s_writer.add_image(
                                f"training/{fname}", rgb, global_step=epoch + 1, dataformats="HWC"
                            )

            if progress:
                sub_bar.close()
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")
            break

        try:
            epoch_val_loss, per_source_val = validation_data(
                model=model,
                loader=validation_dataloader,
                loss_fn=model.loss_fn,
                device=model.target_device,
                verbose=verbose,
                gpu_transforms=train_gpu_transforms if hard_validation else val_gpu_transforms,
                debug=debug is not None and (epoch + 1) % debug == 0,
                visual_debug_path=visual_debug_validation_path,
                s_writer=s_writer if with_tensorboard else None,
                epoch=epoch,
                val_source_boundaries=val_source_boundaries,
            )
            scheduler.step(epoch_val_loss)
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")
            break

        epoch_train_loss = cumulative_train_loss / len(train_dataloader)
        if verbose:
            print(
                f"Epoch {epoch + 1}: train_loss={epoch_train_loss}, val_loss={epoch_val_loss}"
            )

        if with_tensorboard:
            s_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch + 1)
            s_writer.add_scalar("loss/train", epoch_train_loss, global_step=epoch + 1)
            if validation_dataloader:
                s_writer.add_scalar("loss/validation", epoch_val_loss, global_step=epoch + 1)
            for source_name, source_loss in per_source_val.items():
                s_writer.add_scalar(f"loss/validation/{source_name}", source_loss, global_step=epoch + 1)
            s_writer.add_scalar("diagnostics/pred_min", preds.min().item(), global_step=epoch + 1)
            s_writer.add_scalar("diagnostics/pred_max", preds.max().item(), global_step=epoch + 1)
            s_writer.add_scalar("diagnostics/temperature", model.soft_argmax.temperature.item(), global_step=epoch + 1)

        if checkpoint is not None and (epoch + 1) % checkpoint == 0:
            if verbose:
                print(f"Saving intermediate checkpoint. Epoch {epoch + 1} of {epochs}")
            save_checkpoint(
                epoch_progress=(epoch, epochs),
                epoch_losses=(epoch_train_loss, epoch_val_loss),
                run_name=run_name,
                out_dir=out_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
    if with_tensorboard:
        s_writer.close()

    print("Saving final checkpoint.")
    save_checkpoint(
        epoch_progress=(epochs, epochs),
        epoch_losses=(epoch_train_loss, epoch_val_loss),
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )



if __name__ == '__main__':
    resnet = visionmodels.resnet34(weights='IMAGENET1K_V1')
    print(resnet)

