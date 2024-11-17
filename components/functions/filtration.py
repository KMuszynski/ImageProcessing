import numpy as np
from PIL import Image

def create_mask(n):

    kernel_size = 2 * n + 1  # For n=2 -> 3x3 mask, n=3 -> 5x5 mask, etc.
    
    mask = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    return mask

def apply_convolution(arr, mask):

    mask_height, mask_width = mask.shape

    pad_height, pad_width = mask_height // 2, mask_width // 2

    # If color
    if arr.ndim == 3:
        padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')
    else:
        # If grayscale
        padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    filtered_image = np.zeros_like(arr)

    # convolution
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            region = padded_image[i:i + mask_height, j:j + mask_width]
            # If color
            if arr.ndim == 3:
                filtered_image[i, j] = np.sum(region * mask, axis=(0, 1))
            else:
                filtered_image[i, j] = np.sum(region * mask)  #for grayscale

    return filtered_image


def universal_filter(arr, param):
    
    mask = create_mask(param)

    # If color
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Apply convolution to each channel separately
        filtered_image = np.zeros_like(arr)
        for c in range(3):  # Apply convolution to R, G, and B channels separately
            filtered_image[:, :, c] = apply_convolution(arr[:, :, c], mask)
    else:
        # If grayscale
        filtered_image = apply_convolution(arr, mask)

    return filtered_image


def optimized_slowpass_filter(arr):
    # Fixed 3x3 averaging mask
    mask = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], dtype=np.float32) / 9

    # If tcolor
    if arr.ndim == 3 and arr.shape[2] == 3:
        filtered_image = np.zeros_like(arr)
        for c in range(3): 
            filtered_image[:, :, c] = apply_convolution(arr[:, :, c], mask)
    else:
        # If grayscale
        filtered_image = apply_convolution(arr, mask)

    return filtered_image