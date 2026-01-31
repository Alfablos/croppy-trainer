import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import tv_tensors

import common
from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from pathlib import Path
import torch.distributed.optim.post_localSGD_optimizer
import tqdm
from loss import loss_from_str
from architecture import Architecture
from data import SmartDocDataset, get_transforms
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, Flatten
from torch.nn import Sequential, Sigmoid, Linear, ReLU, Dropout
from torch.optim import Adam, Optimizer
import torchvision.models as visionmodels
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from common import Device, DEFAULT_WEIGHTS
from loss import PermutationInvariantLoss


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
        dropout: float,
        learning_rate: float,
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
        self.weights = weights

        # test: remove the pooling layer
        # self.model = visionmodels.resnet18(weights=weights, progress=True)
        self.model = visionmodels.resnet18(weights=weights)
        self.model = nn.Sequential(*list(self.model.children())[:-2]) # exclude pooling layer and fully connected

        # Resnet downsamples x32
        if (images_height % 32 != 0) or (images_width % 32 != 0):
            if architecture == Architecture.RESNET:
                raise ValueError(f"Resnet requires images height and width to be divisible by 32! Current values: h = {images_height}, w = {images_width}")
        h_for_layer = images_height / 32
        w_for_layer = images_width / 32
        # 389120 neurons for 1024x768
        flat_size = int(h_for_layer * w_for_layer * 512) # (512 channels is the number of channels the output has before entering in the, replaced, maxpool layer)
        self.fc = Sequential(
            Flatten(),
            Linear(in_features=flat_size, out_features=512), # replaces maxpool
            ReLU(),
            Linear(in_features=512, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=8) # The coordinates (finally!)
        )

        # self.model.fc = Sequential(
        #     Dropout(p=dropout),
        #     # Adding not just one final layer but three, to give the model
        #     # enough parameters since data will be JPEG with degraded quality
        #     # and rotation!
        #     Linear(in_features=512, out_features=256),  # adds non-linearity
        #     ReLU(),
        #     Linear(in_features=256, out_features=64),
        #     ReLU(),
        #     # output = 8 because we have 8 coordinates:
        #     # the document page has 8 coordinates, not 2 like in bounding boxes of object detection because the camera won't be EXACTLY orthogonal)
        #     Linear(in_features=64, out_features=8),
        #     # pure linear regression, no sigmoid, if corners fall outside the image
        #     # handle via software values like x > width
        #     # Sigmoid(),  # between 0 and 1 for each coordinate
        #     # Sigmoid meaning: tell me exactly where I need to crop
        #     # Linear meaning: tell me where the corners should be, I'll take it from here
        # )

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

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
    def from_trained_config(config: dict, device: Device):
        model: CroppyNet = CroppyNet(
            weights=DEFAULT_WEIGHTS,
            loss_fn=loss_from_str(config["loss_fn"]),
            architecture=Architecture.from_str(config["architecture"]),
            target_device=device,
            images_height=config["images_height"],
            images_width=config["images_width"],
            dropout=config["dropout"],
            learning_rate=config["current_learning_rate"]   # use current or target??
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
    visual_debug_path: str,
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
        labels = labels.as_subclass(torch.Tensor)
        labels = (labels / torch.tensor([new_w, new_h], device="cuda")).flatten(start_dim=1)
        # labels = torch.clamp(labels.flatten(start_dim=1), 0.0, 1.0)

        preds = model(images)
        val_loss += loss_fn(preds, labels).item()

        # only dump debug images on last minibatch
        if debug_fn and batch_n == len(loader):
            end = min(10, len(images))
            # debug_fn(purpose=Purpose.VALIDATION, i=images, l=labels, p=preds)
            img_dict = debug_fn(i=images[0:end], l=labels[0:end], p=preds[0:end])
            for fname, data in img_dict.items():
                cv2.imwrite(visual_debug_path + f"/validation_{fname}", data)

    return val_loss / len(loader)


def train(
    model: CroppyNet,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epochs: int,
    out_dir: str,
    train_len: int,  # only to append the information to filename and specs
    hard_validation: bool,
    debug: int | None,
    with_tensorboard: bool = False,
    verbose=False,
    progress=False,
):
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

    visual_debug_training_subdir = "/visual_debug_training"
    visual_debug_validation_subdir = "/visual_debug_validation"
    visual_debug_training_path = f"{out_dir}" + visual_debug_training_subdir
    visual_debug_validation_path = f"{out_dir}" + visual_debug_validation_subdir
    Path(visual_debug_training_path).mkdir(parents=True, exist_ok=True)
    Path(visual_debug_validation_path).mkdir(parents=True, exist_ok=True)

    # THE MODEL MUST BE MOVED TO cTHE RIGHT DEVICE BEFORE INITIALIZING THE OPTIMIZER
    model = model.to(model.target_device.value)
    # TODO: integrate weight_decay in the CLI or config file
    optimizer = Adam(model.parameters(), lr=model.learning_rate, weight_decay=1e-4)
    # implementing learning rate decay!
    # soft for now
    # TODO: integrate in the CLI or config file
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)

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
                # See https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_tv_tensors.html#but-i-want-a-tvtensor-back
                # normalization may be ineffective on Keypoints, need to unwrap the underlying tensor
                labels = labels.as_subclass(torch.Tensor)
                labels = (labels / torch.tensor([new_w, new_h], device="cuda")).flatten(start_dim=1)
                # No clamping, situations like x > w will be handled post-prediction
                # labels = torch.clamp(labels.flatten(start_dim=1), 0.0, 1.0)

                optimizer.zero_grad()
                preds = model(images)
                # print(f"Preds shape: {preds.shape}")
                # print(f"Labels shape: {labels.shape}")
                loss = model.loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                cumulative_train_loss += loss.item()

                if progress:
                    sub_bar.update(1)

                # debug dump only on LAST minibatch of each epoch if epoch % debug == 0
                if debug and (epoch + 1) % debug == 0 and batch_n == len(train_dataloader):
                    end = min(10, len(images))
                    # debug_fn(i=images[0:end], l=labels[0:end], p=preds[0:end], purpose=Purpose.TRAINING)
                    img_dict = debug_fn(
                        i=images[0:end], l=labels[0:end], p=preds[0:end]
                    )
                    for fname, data in img_dict.items():
                        cv2.imwrite(
                            f"{visual_debug_training_path}/training_{fname}", data
                        )

            if progress:
                sub_bar.close()
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")
            break

        try:
            epoch_val_loss = validation_data(
                model=model,
                loader=validation_dataloader,
                loss_fn=model.loss_fn,
                device=model.target_device,
                verbose=verbose,
                hard=hard_validation,
                debug_fn=debug_fn if debug and (epoch + 1) % debug == 0 else None,
                visual_debug_path=visual_debug_validation_path,
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
            s_writer.add_scalar(
                tag=f"LR_epoch{run_name}", scalar_value=optimizer.param_groups[0]["lr"],
                global_step=epoch + 1
            )
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
            "dropout": model.dropout,
            "initial_learning_rate": model.learning_rate,
            "current_learning_rate": optimizer.param_groups[0]["lr"]
        }
        torch.save(checkpoint, checkpoint_file)
    if with_tensorboard:
        s_writer.close()
