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


def ll_operator(arr, alpha=1.0):
    laplacian = apply_laplacian(arr)

    ll_image = arr + alpha * laplacian

    # Clip values to keep the pixel values within valid range [0, 255]
    ll_image = np.clip(ll_image, 0, 255)

    return ll_image


# Class activity
def apply_roberts(arr):

    arr = arr.astype(np.float32)

    rob = np.zeros_like(arr)

    if arr.ndim == 3:
        for c in range(arr.shape[2]):
            rob[:-1, :-1, c] = np.sqrt(
                # top-left to btm-right
                (arr[:-1, :-1, c] - arr[1:, 1:, c])**2 + (arr[:-1, 1:, c] - arr[1:, :-1, c])**2
            )
    else:
        rob[:-1, :-1] = np.sqrt(
            (arr[:-1, :-1] - arr[1:, 1:])**2 + (arr[:-1, 1:] - arr[1:, :-1])**2
        )

    return rob
