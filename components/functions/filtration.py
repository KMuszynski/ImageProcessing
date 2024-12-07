import numpy as np
import time

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
    start_time = time.time()  # Start timing
    
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

    end_time = time.time()  # End timing
    print(f"universal_filter execution time: {end_time - start_time:.6f} seconds")
    
    return filtered_image


def optimized_slowpass_filter(arr):
    start_time = time.time()
    
    # Predefined 3x3 low-pass filter mask (average filter)
    low_pass_mask = np.ones((3, 3), dtype=np.float32) / 9
    
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    
    height, width = arr.shape
    result = np.zeros((height - 2, width - 2))

    # Apply the convolution operation without extra padding
    for i in range(height - 2):
        for j in range(width - 2):
            result[i, j] = (
                arr[i, j] * low_pass_mask[0, 0] + 
                arr[i, j + 1] * low_pass_mask[0, 1] + 
                arr[i, j + 2] * low_pass_mask[0, 2] +
                arr[i + 1, j] * low_pass_mask[1, 0] + 
                arr[i + 1, j + 1] * low_pass_mask[1, 1] + 
                arr[i + 1, j + 2] * low_pass_mask[1, 2] +
                arr[i + 2, j] * low_pass_mask[2, 0] + 
                arr[i + 2, j + 1] * low_pass_mask[2, 1] + 
                arr[i + 2, j + 2] * low_pass_mask[2, 2]
            )
    
    result = np.clip(result, 0, 255)
    
    end_time = time.time()
    print(f"optimized_slowpass_filter execution time: {end_time - start_time:.6f} seconds")
    
    return result