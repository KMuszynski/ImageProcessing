import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def apply_roberts(arr):
    # Roberts Operator I

    arr = arr.astype(np.float32)

    # Initialize result array
    g = np.zeros_like(arr)

    if arr.ndim == 3:  # for color
        for c in range(arr.shape[2]):
            g[:-1, :-1, c] = np.sqrt(
                (arr[:-1, :-1, c] - arr[1:, 1:, c]) ** 2 +
                (arr[:-1, 1:, c] - arr[1:, :-1, c]) ** 2
            )
    else:  # for grayscale
        g[:-1, :-1] = np.sqrt(
            (arr[:-1, :-1] - arr[1:, 1:]) ** 2 +
            (arr[:-1, 1:] - arr[1:, :-1]) ** 2
        )

    g = np.clip(g, 0, 255)

    return g
