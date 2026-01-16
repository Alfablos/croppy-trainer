import cv2
import numpy as np
import torch
from numpy import ndarray

class MalformedImageError(ValueError):
    def __init__(self):
        super().__init__()




def coords_from_segmentation_mask(mask: ndarray):
    """
        Computes the coordinates of a PERFECTLY RECTANGULARE/SQUARED
        mask which can also be rotated.
        Parameters:
            :parameter mask: a ndarray of pixels (normalized, min 0 - max 1)
        :returns points: a torch tensor of 8 NORMALIZED points
    """
    if np.any(mask > 1) or np.any(mask < 0):
        print("Pixels cannot have values higher that 1 and less than 0. Did you normalize the labels?")
        raise MalformedImageError

    w, h = mask.shape

    # not using mask > 0 (masks for the dataset only have black or white pixels) because if cv2 applies any filters
    # that make some white pixel go toward "black" they'd be counted as black
    # 127 gives 126 of such filter tolerance without altering the result (losing precision)
    white_ys, white_xs = np.where(mask > 0.5) # this returns the INDEXES! Normalization is NEEDED!

    # this returns a (n_points, 2) matrix, each point has an x and a y column
    points = np.column_stack((white_xs, white_ys)) # I love this!


    if len(points) == 0:
        raise MalformedImageError

    # always starting top-left

    # (x + y), returns a (n_points, 1) matrix/vector => the top-left to bottom-right diagonal
    # (y - x) instead returns the orthogonal diagonal: small (very negative) numbers are where y is small and x is big (top-right corner)
    # for each i in points topleft_to_bottoright_diagonal[i] has the sum of its coordinates
    topleft_to_bottoright_diagonal = np.sum(points, axis=1)
    topright_to_bottomleft_diagonal = np.diff(points, axis=1) # diff(a, b) = b - a
    # These two diagonal have the min value to the top (y = 0)!

    # Find the indices of the extremes
    tl = points[np.argmin(topleft_to_bottoright_diagonal)]    # Smallest x + y
    tr = points[np.argmin(topright_to_bottomleft_diagonal)]   # Smallest y - x
    br = points[np.argmax(topleft_to_bottoright_diagonal)]    # Largest x + y
    bl = points[np.argmax(topright_to_bottomleft_diagonal)]   # Largest y - x


    # h and w are, for example 512 and 512
    # tl may be [691, 23]
    # norm_tl = 0.2 / 512
    norm_tl = [tl[0] / w, tl[1] / h]
    norm_tr = [tr[0] / w, tr[1] / h]
    norm_br = [br[0] / w, br[1] / h]
    norm_bl = [bl[0] / w, bl[1] / h]

    return torch.tensor(np.array([
        norm_tl,
        norm_tr,
        norm_br,
        norm_bl
    ]).flatten(), dtype=torch.float32)



if __name__ == "__main__":
    image = cv2.imread(
        filename='/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train/datasheet/0_gt.png',
        flags=cv2.IMREAD_GRAYSCALE
    )
    image = np.divide(image, 255.0)

    whites = coords_from_segmentation_mask(image)

    print(whites)


