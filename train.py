import torchvision.tv_tensors
from torchvision import tv_tensors

import common
from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from pathlib import Path
import torch.distributed.optim.post_localSGD_optimizer
import tqdm
from common import Precision, loss_from_str, Purpose
from architecture import Architecture
from data import SmartDocDataset, get_transforms
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.nn import Sequential, Sigmoid, Linear, ReLU, Dropout
from torch.optim import Adam, Optimizer
import torchvision.models as visionmodels
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from common import Device, DEFAULT_WEIGHTS


class CroppyNet(
    nn.Module
):  # TODO: make it architecture-agnostic: take in base_model and fully_connected_layers to set self.model.fc
    def __init__(
        self,
        weights,
        architecture: Architecture,
        loss_fn: Callable,
        target_device: Device,
        images_height: int,
        images_width: int,
        learning_rate: float = 0.001,
        dropout=0.3,
    ):
        super().__init__()

        self.weights = weights
        self.architecture = architecture
        self.loss_fn = loss_fn
        self.target_device = target_device
        self.images_height: int = images_height
        self.images_width = images_width
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = visionmodels.resnet18(weights=weights, progress=True)
        self.model.fc = Sequential(
            Dropout(p=dropout),
            # Adding not just one final layer but two, to give the model
            # enough parameters since data will be JPEG with degraded quality
            Linear(in_features=512, out_features=128),  # adds non-linearity
            ReLU(),
            # output = 8 because we have 8 coordinates:
            # the document page has 8 coordinates, not 2 like in bounding boxes of object detection because the camera won't be EXACTLY orthogonal)
            Linear(in_features=128, out_features=8),
            Sigmoid(),  # between 0 and 1 for each coordinate
        )
        self.weights = weights

    def forward(self, x):
        return self.model(x)

    def loss_function(self):
        if isinstance(self.loss_fn, L1Loss):
            return "L1Loss"
        if isinstance(self.loss_fn, MSELoss):
            return "MSELoss"
        else:
            raise UnimplementedError

    @staticmethod
    def from_trained_config(config: dict, device: Device):
        model: CroppyNet = CroppyNet(
            weights=DEFAULT_WEIGHTS,
            loss_fn=loss_from_str(config["loss_fn"]),
            architecture=Architecture.from_str(config["architecture"]),
            target_device=device,
            images_height=config["images_height"],
            images_width=config["images_width"],
        )
        model.load_state_dict(config["model_state_dict"])
        return model.to(device.value)  # adds a validation step


@torch.no_grad()
def validation_data(
    model,
    loader,
    loss_fn,
    device: Device,
    verbose: bool,
    hard: bool,
    debug_fn: Callable | None,
) -> float:
    model.eval()
    val_loss = 0.0
    gpu_transforms = get_transforms(common.DEFAULT_WEIGHTS, Device.CUDA, train=hard).to(
        "cuda"
    )
    batch_n = 0

    for images, labels in loader:
        batch_n += 1
        if verbose:
            print(f"Training: tarting batch {batch_n + 1} of {len(loader)}")
        images, labels = images.to(device.value), labels.to(device.value)
        h, w = images.shape[-2:]
        labels_wrapped = tv_tensors.KeyPoints(
            labels.to("cuda"), canvas_size=(h, w), dtype=torch.float32
        )

        images, labels = gpu_transforms(images.to("cuda"), labels_wrapped)
        new_h, new_w = images.shape[-2:]
        labels = labels / torch.tensor([new_w, new_h], device="cuda")
        labels = torch.clamp(labels.flatten(start_dim=1), 0.0, 1.0)

        preds = model(images)
        val_loss += loss_fn(preds, labels).item()
        if debug_fn:
            debug_fn(purpose=Purpose.VALIDATION, i=images, l=labels, p=preds)

    return val_loss / len(loader)


def train(
    model: CroppyNet,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
    epochs: int,
    out_dir: str,
    train_len: int,  # only to append the information to filename and specs
    hard_validation: bool,
    debug: int | None,
    with_tensorboard: bool = False,
    verbose=False,
    progress=False,
):
    if train_dataloader is None:
        raise ValueError("A `train_dataloader` must be specified!")
    if epochs is None:
        raise ValueError("A value for `epochs` must be specified!")

    out_dir = out_dir.rstrip("/")
    if with_tensorboard:
        tensorboard_logdir = out_dir + "/runs"

    if verbose:
        print("Starting training with parameters:")
        for k, v in locals().items():
            print(f"==> {k}: {v}")
            print()

    run_name = f"{model.architecture}_{model.loss_function()}_{model.dropout}dropout_{model.learning_rate}lr_{epochs}epochs_{train_len}x{model.images_height}x{model.images_width}"
    print(f"Starting run {run_name}")
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_files = next(out_dir.walk())[2]
    for f in out_dir_files:
        if f.startswith(run_name) and f.endswith(".pth"):
            raise FileExistsError(
                f"Refusing to overwrite files of a previous run. {f} already exists."
            )

    if with_tensorboard:
        s_writer = SummaryWriter(log_dir=tensorboard_logdir)
        url = utils.launch_tensorboard(tensorboard_logdir)
        if verbose:
            print(f"Tensorboard is listening at {url}")

    if progress:
        epochs_iter = tqdm.trange(epochs, position=0)
    else:
        epochs_iter = range(epochs)

    # THE MODEL MUST BE MOVED TO cTHE RIGHT DEVICE BEFORE INITIALIZING THE OPTIMIZER
    model = model.to(model.target_device.value)
    optimizer = Adam(model.parameters(), lr=model.learning_rate)

    for epoch in epochs_iter:
        model.train()

        if verbose:
            print(f"Starting epoch {epoch + 1}.")

        cumulative_train_loss = 0.0

        debug_fn = (
            lambda purpose, i, l, p: utils.dump_training_batch(
                images=i,
                labels=l,
                preds=p,
                epoch=epoch,
                batch_idx=batch_n,
                purpose=purpose,
                output_dir=f"{out_dir}/visual_debug_{purpose}",
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
                        f"Training: starting batch {batch_n + 1} of {len(train_dataloader)}"
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
                    gpu_transforms = get_transforms(
                        common.DEFAULT_WEIGHTS, Device.CUDA, train=True
                    ).to("cuda")
                    images, labels = gpu_transforms(images.to("cuda"), labels_wrapped)
                new_h, new_w = images.shape[-2:]
                labels = labels / torch.tensor([new_w, new_h], device="cuda")
                labels = torch.clamp(labels.flatten(start_dim=1), 0.0, 1.0)

                optimizer.zero_grad()
                preds = model(images)
                loss = model.loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                cumulative_train_loss += loss.item()

                if progress:
                    sub_bar.update(1)

                if debug and epoch % debug == 0:
                    debug_fn(i=images, l=labels, p=preds, purpose=Purpose.TRAINING)

            if progress:
                sub_bar.close()
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")
            break

        if validation_dataloader:
            try:
                epoch_val_loss = validation_data(
                    model=model,
                    loader=validation_dataloader,
                    loss_fn=model.loss_fn,
                    device=model.target_device,
                    verbose=verbose,
                    hard=hard_validation,
                    debug_fn=debug_fn if debug and epoch % debug == 0 else None,
                )
            except KeyboardInterrupt:
                print("Aborting due to user interruption...")
                break

        epoch_train_loss = cumulative_train_loss / len(train_dataloader)
        if verbose:
            if validation_dataloader is not None:
                print(
                    f"Epoch {epoch + 1}: train_loss={epoch_train_loss}, val_loss={epoch_val_loss}"
                )
            else:
                print(f"Epoch {epoch + 1}: train_loss={epoch_train_loss}")

        if with_tensorboard:
            board_payload = {"train": epoch_train_loss}
            if validation_dataloader:
                board_payload["validation"] = epoch_val_loss
            s_writer.add_scalars(
                main_tag=f"LOSSES_{run_name}",
                tag_scalar_dict=board_payload,
                global_step=epoch + 1,
            )

        # Saving checkpoint
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
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        }
        torch.save(checkpoint, checkpoint_file)
    if with_tensorboard:
        s_writer.close()
