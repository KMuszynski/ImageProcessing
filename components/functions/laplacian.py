import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def apply_laplacian(arr):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)

    # If color
    if arr.ndim == 3:
        laplacian = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            laplacian[:, :, c] = convolve(arr[:, :, c], laplacian_kernel)
    else:
        laplacian = convolve(arr, laplacian_kernel)

    return laplacian
