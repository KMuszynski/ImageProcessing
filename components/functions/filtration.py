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
    
    # Predefined 3x3 low-pass filter mask
    low_pass_mask = np.ones((3, 3), dtype=np.float32) / 9
    pad_height, pad_width = 1, 1  # Padding for a 3x3 mask

    # Add edge padding
    if arr.ndim == 3:  # Color image
        height, width, _ = arr.shape
        padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')
        filtered_image = np.zeros_like(arr, dtype=np.float32)
        for c in range(3):  # Process each channel
            filtered_image[:, :, c] = apply_optimized_convolution(padded_image[:, :, c], low_pass_mask)
    else:  # Grayscale image
        height, width = arr.shape
        padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        filtered_image = apply_optimized_convolution(padded_image, low_pass_mask)

    # Clip the values to valid range
    filtered_image = np.clip(filtered_image, 0, 255)

    end_time = time.time()
    print(f"optimized_slowpass_filter execution time: {end_time - start_time:.6f} seconds")
    return filtered_image.astype(np.uint8)

def apply_optimized_convolution(arr, mask):
    mask_height, mask_width = mask.shape
    height, width = arr.shape
    result = np.zeros((height - mask_height + 1, width - mask_width + 1), dtype=np.float32)

    # Cache the row sums
    row_sums = np.zeros((height - mask_height + 1, width), dtype=np.float32)

    # Precompute row-wise sums for the 3x3 filter
    for i in range(height - 2):  # Traverse rows
        for j in range(width):  # Traverse columns
            row_sums[i, j] = (
                arr[i, j] * mask[0, 0] +
                arr[i + 1, j] * mask[1, 0] +
                arr[i + 2, j] * mask[2, 0]
            )

    # Compute the result using row_sums
    for i in range(height - mask_height + 1):  # Traverse rows
        for j in range(width - mask_width + 1):  # Traverse columns
            result[i, j] = (
                row_sums[i, j] +
                row_sums[i, j + 1] +
                row_sums[i, j + 2]
            )

    return result