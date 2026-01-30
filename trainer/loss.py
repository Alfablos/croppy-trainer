import torch
from torch import nn
from torch.nn import SmoothL1Loss, L1Loss, MSELoss


# When an image is rotated of more than a certain angle (that depends
# on the image h and w) for example counterclockwise, the former top-right
# corner may become the top-left corner! Original image [tl, tr, br, bl]: [tl[0, 0], tr[100, 0], br[100, 100], bl[0, 100]]
# Ground truth after 90° rotation: [[100, 0] was tr now tl, [100, 100] was br now tr, [0, 100] was bl now br, [0, 0] was tl now bl]
# The model is unaware of the rotation and predicts [tl[0, 0], tr[100, 0], br[100, 100], bl[0, 100]]
# Without this normalization the model is severely punished because it predicted a matrix of points that
# in which tl is very distant from the rotated tl because it didn't know it was the tl, it's NOW the tl and
# if what the model thinks it's tl and is very close to a point (whatever it is, tl, tr, br, bl, it depends on the rotation!)
# we CANNOT punish it because it actually predicted a right point! The model, in this context, doesn't have to predict how much the image was rotated,
# it has to predict where 4 points should be in a given image.
# we compute 4 ground truth matrices: 1 for each roll (1 roll = tl becomes bl, tr becomes tl, br becomes tr, bl becomes br; they move in counterclockwise circle)
# The message from this class is: who cares what corner the model thinks this is, if it's close to its prediction we don't punish it!
# We compute the 1 loss between the model prediction and each new position of the points because, we then take as final los the LOWEST
# to emphasize that we don't case about the corner's original role (tl, tr, br, bl) we care about how close was the model to PREDICT
# 4 CORRECT POINTS
class PermutationInvariantLoss(nn.Module):
    """
    PermutationInvariantLoss is a wrapper around existing loss functions
    that prevents loss alterations in case of image rotation.
    It computes 4 losses (each corner might have ended up in place of another),
    but returns the lowest
    """
    def __init__(self, loss_fn: nn.Module):
        super().__init__()

        self.base_loss = loss_fn

        # https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641
        # https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641/2
        if getattr(self.base_loss, 'reduction', None) != 'none':
            print(f"WARNING: {self.base_loss} should be initialized with reduction='none'."
                  "An override is needed and being performed now. Be sure to not rely on that value in the downstream code!")
            self.base_loss.reduction = 'none'

    def forward(self, preds, labels):
        """
        :param preds: a batch of predictions of shape (batch_size, 8) or (batch_size, 4 ,2)
        :param labels: a batch of labels of shape (batch_size, 8) or (batch_size, 4 ,2)
        """
        batch_size = preds.shape[0]
        # force (batch_size, 4, 2)
        preds = preds.view(batch_size, 4, 2)
        labels = labels.view(batch_size, 4, 2)
        losses = []

        for i in range(4):
            # tensor([[ 1., 11.],
            #         [ 2., 22.],
            #         [ 3., 33.],
            #         [ 4., 44.]])  <== moved to
            # >>> torch.roll(t, shifts=1) # no dims crosses dimensions
            # tensor([[44.,  1.],
            #         [11.,  2.],
            #         [22.,  3.],
            #         [33.,  4.]])
            # >>> torch.roll(t, shifts=1, dims=0)
            # tensor([[ 4., 44.],  <== here with shifts=1, dims=0
            #         [ 1., 11.],
            #         [ 2., 22.],
            #         [ 3., 33.]])
            rolled_labels = torch.roll(labels, shifts=i, dims=1) # dims 1 because dim 0 is the batch of images (batch_size, 4, 2) is the shape
            elemwise_loss = self.base_loss(preds, rolled_labels)
            # for each image sum the loss of tl, tr, br, bl (corners)
            coords_loss = elemwise_loss.sum(dim=(1, 2)) # total loss per image PER CORNER => (batch_size, 4)
            losses.append(coords_loss)

        # dims = (batch_size, 4) where 4 is the 4 losses of the permutation
        stack_losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(stack_losses, dim=1) # => (batch_size, 1)

        return min_loss.mean()


            # dim 0 => list of losses
    def inner_to_str(self):
        if isinstance(self.base_loss, MSELoss):
            return "MSELoss"
        else:
            return "L1Loss"


def loss_from_str(s: str, **loss_opts):
    s_lower = s.lower()
    invariant = "invariant" in s_lower

    mae_aliases = ["mae", "maeloss", "mae_loss", "l1", "l1loss", "l1_loss"]
    smooth_mae_aliases = ["smooth_mae", "smooth_maeloss", "smooth_mae_loss", "smooth_l1", "smooth_l1loss", "smooth_l1_loss"]
    mse_aliases = ["mse", "mseloss", "mse_loss", "l2", "l2loss", "l2_loss"]
    base_loss = None
    reduction = 'none' if invariant else loss_opts.get('reduction', 'mean')

    if any([alias in s_lower for alias in smooth_mae_aliases]): # must be evaluated before MAE
        base_loss = SmoothL1Loss(**{**loss_opts, 'reduction': 'none'})
    elif any([alias in s_lower for alias in mae_aliases]):
        base_loss = L1Loss(**{**loss_opts, 'reduction': 'none'})
    elif any([alias in s_lower for alias in mse_aliases]):
        base_loss = MSELoss(**{**loss_opts, 'reduction': 'none'})
    else:
        raise ValueError(f"Unknown loss function: {s}")

    if invariant:
        return PermutationInvariantLoss(base_loss)
    else:
        return base_loss
