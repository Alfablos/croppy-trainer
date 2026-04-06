# The Heatmap Approach: From Dense Regression to Spatial Keypoint Detection

## Table of Contents

1. [The Problem With What We Had](#1-the-problem-with-what-we-had)
2. [The Core Idea: Let the Network Point Instead of Calculate](#2-the-core-idea-let-the-network-point-instead-of-calculate)
3. [What Is a Heatmap?](#3-what-is-a-heatmap)
4. [From Heatmap to Coordinates: Soft-Argmax](#4-from-heatmap-to-coordinates-soft-argmax)
5. [The Full Architecture](#5-the-full-architecture)
6. [Why This Fixes Overfitting (The Parameter Argument)](#6-why-this-fixes-overfitting-the-parameter-argument)
7. [How Training Works End-to-End](#7-how-training-works-end-to-end)
8. [The Code, Piece by Piece](#8-the-code-piece-by-piece)
9. [What Changes in Our Pipeline](#9-what-changes-in-our-pipeline)
10. [Hyperparameters and Tuning](#10-hyperparameters-and-tuning)
11. [What Could Go Wrong](#11-what-could-go-wrong)

---

## 1. The Problem With What We Had

Our previous architecture looked like this:

```
Input image (3, 512, 512)
    │
    ▼
Frozen ResNet18 truncated at layer2    ← Feature extraction, pretrained
    │
    ▼
Feature maps (128, 64, 64)             ← 128 channels, 64x64 spatial grid
    │
    ▼
CoordConv: append x,y grids (130, 64, 64)
    │
    ▼
Conv2d(130 → 8, kernel_size=1)        ← Reduce channels
    │
    ▼
Feature maps (8, 64, 64)
    │
    ▼
Flatten()                               ← HERE IS THE PROBLEM
    │
    ▼
Dense vector of 32,768 numbers
    │
    ▼
Linear(32768 → 256)                    ← 8,388,864 parameters!
    │
    ▼
Linear(256 → 8)                        ← Output: 8 corner coordinates
```

### Why this overfits

The `Linear(32768, 256)` layer is a matrix of 32,768 x 256 = **8.4 million weights**.
We have **22,092 training images**. That's ~380 parameters per training sample.

Think of it this way: if you had 380 free parameters to describe each person
in a room of 22,000 people, you could just memorize everyone's face exactly.
That's what the network does — it memorizes the training set instead of learning
general rules.

### Why can't we just "add more data"?

We could, but there's a deeper issue. The `Flatten()` operation **destroys the
spatial structure** of the feature maps. Before flattening, position (10, 15)
on the 64x64 grid corresponds to a specific region of the input image. After
flattening, that information is encoded as "index 10*64+15 = 655 in a vector
of 32,768 numbers." The Linear layer has to learn, from scratch, what each of
those 32,768 indices means spatially. That's learning a lookup table, not a
transferable function.

### Why can't we just pool?

Pooling (like `AdaptiveAvgPool2d(4)`) reduces the 64x64 map to, say, 4x4 by
averaging. This kills the parameter count (4*4*128 = 2,048 inputs, manageable),
but it also kills spatial precision. Each cell in a 4x4 grid represents a 128x128
pixel region of the original image. You can't localize a corner to within a few
pixels when your finest spatial unit is 128 pixels wide.

**We need an approach that keeps spatial resolution WITHOUT creating millions of
parameters.** That's what heatmap regression does.

---

## 2. The Core Idea: Let the Network Point Instead of Calculate

Imagine two ways to answer "where is the top-left corner of this document?":

**Approach A (Dense Regression — what we had):**
"Look at the entire image, think hard, and tell me two precise numbers: x=0.187, y=0.134"

**Approach B (Heatmap — what we're switching to):**
"Here's a 128x128 grid overlaid on the image. Light up the cell where you see the corner."

Approach B is fundamentally easier because:

1. **It's a local question.** "Is there a corner HERE?" only requires looking at a
   small neighborhood, not the entire image. A corner looks the same whether it's
   in the top-left or bottom-right.

2. **It uses convolutions end-to-end.** Convolutions are naturally translation
   equivariant — they apply the same filter everywhere. A convolutional corner
   detector works at all positions automatically. A Dense layer has to learn
   each position independently.

3. **It produces few parameters.** A `Conv2d(64, 4, kernel_size=1)` layer has
   64*4 + 4 = **260 parameters**. Compare that to 8.4 million.

---

## 3. What Is a Heatmap?

A heatmap is a 2D grid (tensor) where each cell contains a value representing
"how strongly I believe the thing I'm looking for is at this location."

In our case, the network outputs **4 heatmaps**, one for each document corner:

```
Heatmap 0: "Where is the top-left corner?"     → shape (128, 128)
Heatmap 1: "Where is the top-right corner?"    → shape (128, 128)
Heatmap 2: "Where is the bottom-right corner?" → shape (128, 128)
Heatmap 3: "Where is the bottom-left corner?"  → shape (128, 128)

Stacked together: (4, 128, 128)
For a batch of B images: (B, 4, 128, 128)
```

A well-trained network produces heatmaps with a sharp peak at the corner location
and low values everywhere else:

```
Heatmap for top-left corner (conceptual, not actual values):

   0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  1  3  5  3  1  0  0  0  0
   0  0  0  0  1  5 12 18 12  5  1  0  0  0
   0  0  0  0  3 12 35 50 35 12  3  0  0  0    ← peak here
   0  0  0  0  1  5 12 18 12  5  1  0  0  0
   0  0  0  0  0  1  3  5  3  1  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0
```

**Important:** We do NOT provide these heatmaps as ground truth. We do NOT tell
the network "this is what your heatmap should look like." The heatmaps are an
**internal representation** that the network discovers on its own. We only
supervise the final coordinates that come out of the soft-argmax (explained next).

The network learns to produce peaked heatmaps because that's what minimizes the
coordinate prediction error.

---

## 4. From Heatmap to Coordinates: Soft-Argmax

### The problem: argmax isn't differentiable

Given a heatmap, the obvious way to extract a coordinate is `argmax` — find the
cell with the highest value:

```python
heatmap = [[0, 0, 0],
           [0, 0, 1],   ← max is here
           [0, 0, 0]]
position = argmax(heatmap)  # → row=1, col=2
```

But `argmax` has **zero gradient everywhere**. Small changes to the heatmap values
don't change which cell is the maximum (until a threshold is crossed, at which point
the gradient is infinite). You can't backpropagate through it, so you can't train
the network end-to-end.

If you've taken Andrew Ng's course, think of it like a step function — its derivative
is 0 almost everywhere and undefined at the step. That's useless for gradient descent.

### The solution: softmax + weighted average = soft-argmax

Soft-argmax is a two-step trick that makes coordinate extraction differentiable:

**Step 1: Turn the heatmap into a probability distribution using softmax.**

`softmax` (which you know from classification) takes a vector of arbitrary numbers
and turns them into a probability distribution (all positive, sums to 1):

```
Raw heatmap values:  [0.1, 0.3, 2.5, 0.2, 0.1]
After softmax:       [0.05, 0.06, 0.55, 0.06, 0.05]
                      ↑ all positive    ↑ biggest gets most probability
                                          sum = ~1.0
```

In PyTorch, `F.softmax(tensor, dim=-1)` does this. The `dim` argument says which
dimension to normalize over. For a (B, 4, 64, 64) heatmap, we first flatten the
spatial dimensions to (B, 4, 4096), apply softmax over the last dimension (the
4096 spatial positions), then reshape back.

**Why softmax and not just dividing by the sum (i.e., normalize)?**
Softmax applies `exp()` first, which makes the distribution sharper — the highest
value gets disproportionately more weight. This helps produce precise coordinates.
It also handles negative values gracefully (exp of a negative is a small positive).

**Step 2: Compute the expected (mean) position.**

Once we have a probability distribution over the 128x128 grid, the coordinate is
simply the **weighted average** of all positions, weighted by their probabilities:

```
Positions:      [0.0,  0.25, 0.5,  0.75, 1.0 ]
Probabilities:  [0.05, 0.06, 0.55, 0.06, 0.05]

x = 0.0*0.05 + 0.25*0.06 + 0.5*0.55 + 0.75*0.06 + 1.0*0.05
x = 0 + 0.015 + 0.275 + 0.045 + 0.05
x = 0.385
```

If the probability is concentrated at position 0.5, the output is close to 0.5.
If the distribution is broader, the output is a weighted average of the spread.

**This is differentiable.** Both softmax and weighted average are smooth, continuous
operations with well-defined gradients. PyTorch's autograd can backpropagate through
them without any issues.

### Sub-pixel precision

This is the magical part. Even though our grid is 128x128, the output coordinates
are continuous floats. If the true corner is between grid cells 60 and 61 (at position
0.477 on a 0-1 scale), the soft-argmax naturally interpolates: the probability
distribution straddles the two cells, and the weighted average lands at 0.477.

The precision is limited only by float32 arithmetic, not by the grid resolution.
In practice, a 128x128 grid with soft-argmax gives precision equivalent to a much
finer grid. (The backbone produces 64x64 features; the head upsamples to 128x128
before producing heatmaps, doubling coordinate precision.)

### How it looks in code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax2D(nn.Module):
    """
    Convert (B, K, H, W) heatmaps to (B, K, 2) coordinates.
    K = number of keypoints (4 for our corners).
    Output coordinates are normalized to [0, 1].

    The learnable temperature scales logits before softmax:
    higher temperature → sharper distribution → more precise coordinates.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, heatmaps):
        B, K, H, W = heatmaps.shape

        # --- Step 1: scale by temperature and softmax over spatial dims ---
        # Reshape (B, K, H, W) → (B, K, H*W) so softmax runs over all positions
        # Multiply by temperature BEFORE softmax to control distribution sharpness
        flat = heatmaps.view(B, K, -1) * self.temperature  # (B, 4, 16384)
        probs = F.softmax(flat, dim=-1)                     # (B, 4, 16384)  sums to 1 per corner
        probs = probs.view(B, K, H, W)                      # (B, 4, 128, 128)

        # --- Step 2: create coordinate grids ---
        # These are fixed 0-to-1 grids, same idea as the CoordConv channels
        x_coords = torch.linspace(0, 1, W, device=heatmaps.device)  # [0.000, 0.008, ..., 1.000]
        y_coords = torch.linspace(0, 1, H, device=heatmaps.device)

        # --- Step 3: weighted average ---
        # For x: sum probabilities along height axis, then dot with x positions
        #   probs.sum(dim=-2) collapses the H dimension → (B, K, W)
        #   Multiplying by x_coords and summing gives the expected x
        x = (probs.sum(dim=-2) * x_coords).sum(dim=-1)   # (B, 4)
        # For y: same but collapse the W dimension first
        y = (probs.sum(dim=-1) * y_coords).sum(dim=-1)   # (B, 4)

        return torch.stack([x, y], dim=-1)                # (B, 4, 2)
```

**Why learnable temperature?** Early in training, the heatmaps are diffuse — the
network doesn't yet know where corners are. A moderate temperature keeps gradients
flowing across the whole grid. As training progresses, the optimizer increases
temperature, making the softmax distribution sharper and concentrating probability
mass on the peak. The network learns both where to peak AND how sharp to make the
peak, jointly optimized by the same loss function.

Let's walk through the `x` computation dimension by dimension:

```
probs shape:            (B, 4, 128, 128)   ← probability at each grid cell
probs.sum(dim=-2):      (B, 4, 128)        ← collapse rows: "how much probability is in each column?"
  * x_coords:           (B, 4, 128)        ← weight each column by its x position
.sum(dim=-1):           (B, 4)             ← sum across columns: expected x value per corner

Same logic for y, but collapsing columns first (dim=-1) then summing rows.
```

**Why `sum(dim=-2)` before multiplying by `x_coords`?**
Think of it as marginalization from probability theory. To find the expected x position,
we need the marginal probability distribution over x (columns). We get that by summing
out y (rows). Then E[x] = sum(P(x) * x) — the expected value formula.

---

## 5. The Full Architecture

```
Input image (3, 512, 512)
    │
    ▼
ResNet18 conv1 + bn1 + relu + maxpool       ← Frozen (universal edge features)
    │
    ▼
ResNet18 layer1                              ← Frozen (low-level patterns)
    │
    ▼
ResNet18 layer2                              ← UNFROZEN — learns corner-relevant features
    │
    ▼
Feature maps (128, 64, 64)                   ← 128 channels, 64x64 spatial grid
    │
    ▼
CoordConv: append x,y grids (130, 64, 64)
    │
    ▼
Conv2d(130 → 64, kernel_size=3, padding=1)   ← Convolutional heatmap head
BatchNorm2d(64)
ReLU
    │
    ▼
Feature maps (64, 64, 64)                    ← Still 64x64 spatial resolution
    │
    ▼
Upsample(scale_factor=2, bilinear)           ← Double spatial resolution
    │
    ▼
Feature maps (64, 128, 128)                  ← Finer grid for precise localization
    │
    ▼
Conv2d(64 → 32, kernel_size=3, padding=1)    ← Refine upsampled features
BatchNorm2d(32)
ReLU
    │
    ▼
Feature maps (32, 128, 128)
    │
    ▼
Conv2d(32 → 4, kernel_size=1)               ← Produce 4 heatmaps
    │
    ▼
Heatmaps (4, 128, 128)                       ← One heatmap per corner
    │
    ▼
SoftArgmax2D (learnable temperature)          ← Differentiable coordinate extraction
    │
    ▼
Coordinates (4, 2)  → flattened to (8,)      ← Same output format as before!
```

### Why unfreeze layer2?

With the backbone fully frozen, we relied entirely on the 75K-param head to map
ImageNet features to document corners. The initial results showed **underfitting**:
loss plateaued at ~0.84 (~54px error) and couldn't improve further, even with LR
already reduced to 1e-5. The frozen features simply weren't discriminative enough
for our task.

The fix: **selective unfreezing**. We keep conv1, bn1, and layer1 frozen — these
detect universal visual primitives (edges, textures) that transfer perfectly. We
unfreeze layer2, which learns mid-level features. In the ImageNet domain, layer2
detects things like "corners of objects" and "texture boundaries" — exactly the
features we need, but tuned for natural images, not documents.

By unfreezing layer2, the network can adapt these mid-level features to detect
document-specific corners: where a white page meets a colored background, where
a page edge creates a specific contrast pattern, etc. The frozen lower layers
still provide the edge/texture building blocks — layer2 just recombines them
for our domain.

This is controlled by two config flags:
```python
freeze_backbone = True          # freeze conv1, bn1, layer1
freeze_backbone_layer2 = False  # unfreeze layer2 so it can adapt
```

### What each layer does

**`Conv2d(130, 64, kernel_size=3, padding=1)`**: A 3x3 convolution. It looks at
each 3x3 neighborhood in the 130-channel feature map and produces 64 output
channels. The `padding=1` ensures the spatial dimensions stay at 64x64
(without padding, a 3x3 conv on a 64x64 input would produce 62x62).

This is the layer that learns "what does a document corner look like in feature
space?" It combines the backbone's visual features (128 channels) with the
coordinate grids (2 channels) to detect corners at each spatial location.

**`Upsample(scale_factor=2, bilinear)`**: Doubles the spatial resolution from
64x64 to 128x128 via bilinear interpolation. This is NOT a learnable layer — it
simply interpolates between existing values. The purpose is to give subsequent
conv layers a finer canvas to work on, which directly improves the precision of
the final soft-argmax coordinate extraction.

**`Conv2d(64, 32, kernel_size=3, padding=1)`**: A second 3x3 convolution that
refines the upsampled features. After bilinear upsampling, the features are
"smooth" — this conv learns to sharpen them into peaked heatmaps. The channel
reduction (64 → 32) concentrates the representation.

**`BatchNorm2d`**: Applied after each conv layer. Normalizes each channel to have
mean~0, std~1, which stabilizes training. Same concept as BatchNorm in Ng's
courses, just applied to 2D feature maps.

**`Conv2d(32, 4, kernel_size=1)`**: A 1x1 convolution, also called a pointwise
convolution. It processes each spatial position independently — like having a tiny
Dense layer at every pixel. It takes the 32 features at each location and
outputs 4 values: one "score" per corner. No spatial mixing happens here.

**`SoftArgmax2D`**: Converts the 4 score maps into 4 (x,y) coordinate pairs.
Has one trainable parameter: **temperature**, which scales the logits before
softmax to control distribution sharpness (see Section 4). The coordinate grids
and softmax computation are fixed differentiable operations.

### Why kernel_size=3 for the first two convs but 1 for the last?

The 3x3 convs need spatial context — to decide "is this a corner?" you need to
look at a small neighborhood, not just one pixel. 3x3 is the smallest useful
receptive field. Having two 3x3 convs (one before and one after upsampling) gives
the head a larger effective receptive field.

The last conv just needs to collapse channels — at each position, combine 32
feature responses into 4 corner scores. No spatial context needed, so 1x1 is
sufficient and cheaper.

### Why upsample inside the head?

The backbone produces features at 64x64 (8x downsampled from 512x512 input).
Soft-argmax extracts coordinates as the expected position over the heatmap grid.
On a 64x64 grid, adjacent cells are 1/64 apart — so even a perfectly peaked
heatmap can only place coordinates to ~1.5% precision (~8px at 512 input).

Upsampling to 128x128 halves this to ~0.8% (~4px at 512). The bilinear upsample
itself adds no parameters — the precision gain comes from giving the subsequent
conv layers and soft-argmax a finer grid to work with.

---

## 6. Why This Fixes Overfitting (The Parameter Argument)

Let's count parameters:

| Layer | Parameters | Trainable? | Calculation |
|-------|-----------|------------|-------------|
| conv1 + bn1 (backbone) | 9,536 | No (frozen) | ResNet stem |
| layer1 (backbone) | 147,968 | No (frozen) | Two BasicBlock modules |
| layer2 (backbone) | 525,568 | **Yes** | Two BasicBlock modules, unfrozen |
| Conv2d(130, 64, k=3, pad=1) | 74,880 | **Yes** | 130 × 64 × 3 × 3 + 64 bias |
| BatchNorm2d(64) | 128 | **Yes** | 64 × 2 (scale + shift) |
| Upsample(×2, bilinear) | 0 | — | No learnable parameters |
| Conv2d(64, 32, k=3, pad=1) | 18,464 | **Yes** | 64 × 32 × 3 × 3 + 32 bias |
| BatchNorm2d(32) | 64 | **Yes** | 32 × 2 (scale + shift) |
| Conv2d(32, 4, k=1) | 132 | **Yes** | 32 × 4 × 1 × 1 + 4 bias |
| SoftArgmax2D temperature | 1 | **Yes** | Scalar learnable parameter |
| **Total trainable** | **~619,200** | | layer2 + head |
| **Total frozen** | **~157,500** | | conv1 + bn1 + layer1 |

Compare:

| Approach | Trainable params | Params per training sample |
|----------|-----------------|---------------------------|
| Previous (Dense head) | ~8,390,000 | ~380 |
| Heatmap head only (frozen backbone) | ~94,000 | ~4.3 |
| **Current (head + unfrozen layer2)** | **~619,200** | **~28** |

A ratio of **~28 parameters per sample** is comfortable — well below the danger
zone of the old dense head (~380), but enough capacity to actually learn the task.

Why is ~27 OK for conv layers when ~380 was deadly for dense? Because convolutional
parameters are **shared across all spatial positions**. A 3×3 filter with 64 outputs
has 576 weights, but those same weights are applied at all 4,096 positions on the
64×64 grid. The effective parameter-to-"decision" ratio is much lower. Dense layers
have no sharing — each weight connects exactly one input to one output, making them
pure lookup tables that memorize training data.

### Where did the parameters go?

The old Dense head needed 8.4M parameters because the `Linear(32768, 256)` layer
had to learn a specific weight for every position in the flattened feature map.
Position 0 is the top-left of channel 0, position 64 is one row down in channel 0,
position 4096 is the top-left of channel 1, etc. Every position gets its own dedicated
weight — there's no sharing.

The new Conv head uses **the same 3x3 filter everywhere on the 64x64 grid**. There
are 64 such filters, each with 130*3*3 = 1,170 weights. The same 1,170 weights are
reused at all 4,096 positions. That's the power of convolutions — parameter sharing.

From Ng's course you know that a convolutional layer's parameter count depends on the
kernel size and channel count, NOT on the spatial dimensions. A Conv2d(130, 64, k=3)
has the same number of parameters whether the input is 64x64 or 640x640. A Linear
layer's parameter count scales linearly with input size. That's why flattening large
feature maps into Linear layers creates parameter explosions.

---

## 7. How Training Works End-to-End

### Forward pass

```
1. Image (3, 512, 512) enters the backbone
2. conv1 + bn1 + layer1 extract low-level features (frozen, don't change)
3. layer2 refines into mid-level features (128, 64, 64) — TRAINABLE
4. CoordConv appends x,y grids → (130, 64, 64)
5. First conv (130→64, 3×3) processes at 64x64
6. Upsample doubles resolution → (64, 128, 128)
7. Second conv (64→32, 3×3) + final conv (32→4, 1×1) → heatmaps (4, 128, 128)
8. SoftArgmax2D (with learnable temperature) extracts coordinates (4, 2) → flattened to (8,)
9. Loss = PermutationInvariantLoss(predicted_coords, ground_truth_coords)
```

### Backward pass (gradient flow)

```
Loss (scalar)
    │  ∂loss/∂predicted_coords
    ▼
Predicted coordinates (8,)
    │  ∂coords/∂heatmaps  ←  Gradient of SoftArgmax2D (differentiable!)
    │  ∂loss/∂temperature  ←  Temperature is also updated
    ▼
Heatmaps (4, 128, 128)
    │  ∂heatmaps/∂conv_weights  ←  Standard conv backprop
    ▼
Conv2d(32→4) + Conv2d(64→32)  ←  Trainable, operates at 128×128
    │
    ▼
Upsample (bilinear — has well-defined gradients, no trainable params)
    │
    ▼
Conv2d(130→64) (trainable, operates at 64×64)
    │
    ▼
CoordConv channels (no gradient — fixed grids, not trainable)
    │
    ▼
layer2 (TRAINABLE — gradients flow through and update these weights)
    │
    ▼
layer1 + conv1 + bn1 (no gradient — frozen)
```

**The gradient of soft-argmax**, intuitively:

If the predicted x coordinate is **too far right** compared to ground truth
(predicted 0.6, truth is 0.4), the gradient tells the heatmap: "shift your
probability mass to the left." The heatmap cells on the left side of the peak
get positive gradients (become stronger) and cells on the right get negative
gradients (become weaker). The peak moves left. Next forward pass, the coordinate
is closer to 0.4.

This is smooth and well-behaved — there are no discontinuities, no vanishing
gradients (softmax saturates less than sigmoid for this use case because it operates
over thousands of values, not just one).

### What the loss function sees

The output is still 8 floats, normalized 0-1, in the same format as before:
`[tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]`

`PermutationInvariantLoss` works unchanged. It still tries all 4 cyclic
permutations and takes the minimum — it doesn't care how the coordinates were
produced.

---

## 8. The Code, Piece by Piece

### SoftArgmax2D (nn.Module in config.py, with learnable temperature)

```python
class SoftArgmax2D(nn.Module):
    """
    Convert heatmaps to normalized coordinates via differentiable soft-argmax.
    Learnable temperature controls distribution sharpness.

    Args:
        temperature: initial temperature value (default 1.0).

    Input:
        heatmaps: (B, K, H, W) raw scores (logits) from the conv head.
                  B = batch size, K = number of keypoints (4 corners).

    Returns:
        (B, K, 2) tensor of (x, y) coordinates, each in [0, 1].
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, heatmaps):
        B, K, H, W = heatmaps.shape

        # Flatten spatial dims, scale by learnable temperature, then softmax.
        # F.softmax with dim=-1 normalizes across the 16384 spatial positions so they sum to 1.
        # Higher temperature → sharper distribution → more precise coordinates.
        flat = heatmaps.view(B, K, -1) * self.temperature  # (B, 4, 16384)
        probs = F.softmax(flat, dim=-1)                     # (B, 4, 16384) — sums to 1 per corner
        probs = probs.view(B, K, H, W)                      # (B, 4, 128, 128)

        # Fixed coordinate grids — these are NOT learned, they're just 0-to-1 rulers.
        x_coords = torch.linspace(0, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        y_coords = torch.linspace(0, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)

        # Marginalize out y to get P(x), then compute E[x] = sum(P(x) * x).
        x = (probs.sum(dim=-2) * x_coords).sum(dim=-1)   # (B, 4)

        # Same thing for y: marginalize out x (columns), weight by y positions.
        y = (probs.sum(dim=-1) * y_coords).sum(dim=-1)   # (B, 4)

        return torch.stack([x, y], dim=-1)                # (B, 4, 2)
```

### The new head (replaces the Dense head in config.py)

```python
head = nn.Sequential(
    nn.Conv2d(head_input_channels, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64×64 → 128×128
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 4, kernel_size=1),  # 4 heatmaps, one per corner
)
```

**Why upsample?** The backbone produces 64×64 features. Soft-argmax precision is
limited by grid resolution — on 64×64, adjacent cells are 1/64 apart (~8px at 512
input). Upsampling to 128×128 doubles precision for free (bilinear interpolation
has no learnable parameters). The subsequent 3×3 conv then learns to sharpen the
interpolated features into peaked heatmaps.

**Why no Dropout?** Dropout randomly zeroes out neurons during training to prevent
co-adaptation. In convolutional layers, spatial dropout (zeroing entire channels)
is the equivalent, but with only ~94K head parameters and 22K+ samples, we're not
in the overfitting danger zone. If we see signs of overfitting, we can add
`nn.Dropout2d(p=0.1)` after a ReLU — but start without it.

**Why no Flatten, no Linear?** That's the whole point. The spatial structure is
preserved all the way through. The heatmaps ARE the spatial output. SoftArgmax2D
reads off the coordinates. No flattening needed.

### Changes to the forward method (train.py)

```python
def forward(self, x):
    x = self.model(x)                          # backbone: (B, 128, 64, 64)
    if config.coord_conv:
        x = self.add_coords(x)                  # coord grids: (B, 130, 64, 64)
    heatmaps = self.fc(x)                       # conv head: (B, 4, 128, 128)
    coords = self.soft_argmax(heatmaps)          # differentiable: (B, 4, 2)
    return coords.flatten(start_dim=1)           # (B, 8)
```

Note that `self.soft_argmax` is a `SoftArgmax2D` instance (an `nn.Module`) rather
than a standalone function. This ensures the learnable temperature parameter is
registered in the model, saved in checkpoints, and updated by the optimizer.

The output is `(B, 8)` just like before. Everything downstream — loss function,
TensorBoard logging, checkpoint saving, debug visualization — works unchanged.

### Selective backbone unfreezing (train.py `__init__`)

```python
if config.freeze_backbone:
    for param in self.model.parameters():
        param.requires_grad = False
    # Selectively unfreeze layer2 (index 5 in the Sequential)
    if not config.freeze_backbone_layer2:
        for param in self.model[5].parameters():
            param.requires_grad = True
```

**Why `self.model[5]`?** The backbone is built as:
```python
backbone_layers = list(base.children())[:6]  # conv1, bn1, relu, maxpool, layer1, layer2
self.model = nn.Sequential(*backbone_layers)
```
So index 5 is layer2. We freeze everything first, then selectively unfreeze layer2.

---

## 9. What Changes in Our Pipeline

### What stays the same

- **Data pipeline**: `SmartDocDataset`, Arrow stores, precomputed images — untouched.
- **Ground truth format**: 8 floats (corner coordinates). No new labels.
- **Loss function**: `PermutationInvariantLoss` with L1 inside. Same as before.
- **Transforms**: CPU and GPU augmentations. Same as before.
- **Label normalization**: Dividing by image dimensions to get 0-1 range. Same as before.
- **Optimizer**: Adam. Same as before (but now only ~94K head params + ~526K layer2 to optimize).
- **Scheduler**: ReduceLROnPlateau. Same as before.
- **Checkpointing, TensorBoard, debug images**: All work on the (B, 8) output, unchanged.

### What changes

| Component | Before | After |
|-----------|--------|-------|
| `config.py`: head | Conv2d→Flatten→Linear(32768,256)→Linear(256,8) | Conv2d(130,64)→Upsample(×2)→Conv2d(64,32)→Conv2d(32,4) |
| `config.py`: new module | — | `SoftArgmax2D` (learnable temperature) |
| `config.py`: backbone freezing | Fully frozen | layer2 unfrozen |
| `train.py`: forward() | `return self.fc(x)` | `heatmaps = self.fc(x)` then `self.soft_argmax(heatmaps).flatten(1)` |
| Trainable parameters | ~8.4M (dense) | ~619K (layer2 + conv head + temperature) |

### What about CoordConv?

CoordConv still makes sense here — arguably more than before. The 3x3 conv in the
head now combines visual features (from the backbone) with position information
(from the coord grids) at each spatial location. The filter can learn patterns like
"strong vertical edge + near the left side of the image → likely left corner."
Without the coord grids, the filter only knows about visual features and can't
distinguish left corners from right corners based on position.

Whether it actually helps is an empirical question. It's cheap (2 extra channels,
no extra parameters), so we keep it and can ablate it later (set `coord_conv = False`
and change `head_input_channels` to 128).

### Resume training

Checkpoints now save the full training state: model weights, optimizer state (Adam
momentum estimates), and LR scheduler state (patience counter, best metric). This
means training can be resumed from any checkpoint without losing progress:

```bash
python croppy.py train ... --resume path/to/checkpoint.pth --epochs 200
```

When resuming:
- The model, optimizer, and scheduler are restored from the checkpoint
- Training continues from the saved epoch (e.g., checkpoint at epoch 50 → trains 51–200)
- The output directory check is skipped (we're adding to an existing run)
- The run name stays the same, so TensorBoard shows a continuous curve

Without scheduler state restoration, `ReduceLROnPlateau` would reset its patience
counter, potentially keeping the LR too high after a resume. Without optimizer state,
Adam would lose its per-parameter momentum estimates and produce noisy updates for
~10 epochs.

### Mixed datasets and merge-stores

The training pipeline supports multiple data sources per purpose (training/validation).
Currently configured:

- **SmartDoc Extended**: Synthetic — documents digitally pasted onto backgrounds
- **SmartDoc 2015 Original**: Real photos of documents on tables

The `--limit` flag during precompute applies per-source: `--limit 2500` gives 2500
from each source = 5000 total.

For building stores incrementally (precompute sources separately, then combine):

```bash
# Precompute each source into its own store, then merge:
python croppy.py merge-stores store_a.arrow store_b.arrow -o merged.arrow
```

The `merge-stores` command validates that all input stores have matching dimensions
(h, w) before concatenating.

---

## 10. Hyperparameters and Tuning

### Temperature (learnable, implemented)

The `SoftArgmax2D` module has a learnable temperature parameter (initialized at 1.0)
that scales logits before softmax. The convention is **multiplication** rather than
division:

```python
flat = heatmaps.view(B, K, -1) * self.temperature  # scale logits
probs = F.softmax(flat, dim=-1)
```

- **temperature > 1**: Sharper distribution → more precise coordinates. The network
  learns to increase temperature as heatmaps become more peaked.
- **temperature = 1**: Neutral (standard softmax). This is the initialization.
- **temperature < 1**: Smoother distribution → less precise but easier to train.

The optimizer adjusts temperature jointly with all other parameters. In TensorBoard,
watch `diagnostics/temperature` to track its evolution. If temperature grows very
large (>50), the softmax is approaching hard argmax and gradients may vanish —
but in practice the loss provides a natural brake.

### Head depth and upsampling

Our head has 3 conv layers with bilinear upsampling between the first and second:

- **Conv2d(130→64, k=3)** at 64×64: Detects corner features from backbone + coords.
- **Upsample(×2, bilinear)**: Free precision gain (64×64 → 128×128).
- **Conv2d(64→32, k=3)** at 128×128: Refines upsampled features into sharp peaks.
- **Conv2d(32→4, k=1)** at 128×128: Produces 4 heatmaps.

Total head params: ~94K. The upsampling step is key — it doubles coordinate
precision without adding learnable parameters. The second 3×3 conv operates at
the finer resolution and can learn to produce sharper heatmaps than would be
possible at 64×64.

### Learning rate

With the unfrozen layer2 and full dataset, LR = 0.0001 works well. The
`ReduceLROnPlateau` scheduler (factor=0.5, patience=4, threshold=1e-3) reduces it
when validation loss stalls. With the previous fully-frozen backbone, we started
at 0.001 but the scheduler reduced it to 1e-5 quickly as the model hit its
capacity ceiling. The larger dataset required dropping to 1e-4 to avoid oscillation.

### Gradient clipping

Still useful as a safety net. `grad_clip_max_norm=1.0` is fine. With the unfrozen
layer2, gradient norms can be larger than with a frozen backbone (gradients flow
further), so clipping prevents occasional instability.

### Backbone unfreezing strategy

The decision of what to freeze/unfreeze follows a diagnostic flowchart:

1. **Start with fully frozen backbone + heatmap head** (~94K params)
2. If **underfitting** (training loss plateaus high, LR already tiny):
   → Unfreeze layer2 for more capacity
3. If still underfitting → add a deeper head with upsampling ← **this is where we are now**
4. If **overfitting** (val loss rises while train loss drops):
   → Freeze layer2 back, try deeper head instead (more capacity but with conv
     parameter sharing, which resists memorization better than unfreezing backbone
     layers that have more per-position expressiveness)

---

## 11. What Could Go Wrong

### Problem: The network predicts the mean of all corners

If the heatmaps are very broad and flat (uniform distribution), soft-argmax returns
(0.5, 0.5) for everything — the center of the image. This can happen early in
training before the network has learned anything.

**Why it's OK:** The gradient of soft-argmax through a flat distribution is nonzero
and well-defined. It pushes the network to concentrate probability mass at the
correct locations. The uniform-distribution start is equivalent to "I don't know
yet" and the network learns from there. If training seems stuck at center predictions
for many epochs, try increasing the learning rate.

### Problem: Two corners are close together

If two corners are very close (e.g., a nearly-folded document), their heatmap peaks
might merge. Soft-argmax of a bimodal distribution returns the midpoint between the
two modes, not either mode.

**Why it's OK for us:** Each corner has its OWN heatmap (4 separate channels).
They don't interfere with each other. Corner 0's heatmap only needs one peak.
This would only be a problem if we had a single heatmap for all corners.

### Problem: Corner falls outside the 64x64 grid bounds

Our coordinates are normalized to [0, 1], which maps to the full 512x512 image.
The 64x64 grid also spans [0, 1]. A corner at position 0.02 (near the edge)
lands on grid cell ~1.3. The soft-argmax can still represent this — it's just
a weighted average that lands near 0.02. No issues as long as the corner is
within the image.

### Problem: Loss doesn't decrease at all

If the conv head produces constant outputs (all zeros from bad initialization),
soft-argmax returns (0.5, 0.5) for all corners. The loss is then constant and
gradients might be too small to escape.

**Mitigation:** This is rare with standard PyTorch initialization (Kaiming/Xavier),
which produces varied outputs from the start. BatchNorm also helps by ensuring
the distribution has reasonable variance. If it happens, check that the backbone
is actually producing non-constant features (it should be, it's pretrained and
not modified).

---

## Summary

The heatmap approach replaces the **flatten + dense** bottleneck with a **fully
convolutional head + differentiable soft-argmax**, plus selective backbone unfreezing:

```
BEFORE: frozen backbone → flatten → 8.4M param linear → 8 coords  ← overfits
AFTER:  partially-unfrozen backbone → 75K param conv head
          → 4 heatmaps → soft-argmax → 8 coords                    ← generalizes
```

Total trainable: ~600K params (525K from unfrozen layer2 + 75K head).
Total frozen: ~157K params (conv1 + bn1 + layer1).

Ground truth stays the same (8 coordinates). Loss stays the same. Data pipeline
stays the same. The changes are:
1. Dense head → convolutional heatmap head with soft-argmax
2. Fully frozen backbone → selective layer2 unfreezing
3. Resume training from checkpoints (optimizer + scheduler state)
4. `merge-stores` command for incremental dataset building

The key idea: instead of asking a giant Linear layer to memorize the mapping from
32,768 feature positions to 8 numbers, we ask a small Conv layer to detect corners
locally, and then read off their positions with a weighted average. The Conv layer
learns one universal corner detector (with 3x3 spatial context and position
awareness from CoordConv). The Linear layer had to learn 32,768 independent weights.
That's the difference between learning and memorizing.
