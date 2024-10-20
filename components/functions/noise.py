# components/functions/noise.py
import numpy as np


def doMedianFilter(param, arr):
    print("Function doMedianFilter invoked with param: " + param)

    param = int(param)
    arr = np.array(arr)

    if arr.ndim == 2:  # Grayscale image
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:  # Color image
        height, width, channels = arr.shape
    else:
        raise ValueError("Unexpected array shape. Please check the image format.")

    # Padding size for the kernel (param x param window)
    pad_size = param // 2

    # Empty array to store results
    result_arr = np.zeros_like(arr)

    for c in range(channels):
        if channels == 1:
            padded_arr = np.pad(arr, pad_size, mode='constant', constant_values=0)
        else:
            padded_arr = np.pad(arr[:, :, c], pad_size, mode='constant', constant_values=0)

        # Apply median filter
        for i in range(height):
            for j in range(width):
                # Extract the neighborhood window
                window = padded_arr[i:i + param, j:j + param].flatten()

                # Sort the values in the window and find the median
                median_value = np.median(sorted(window))

                # Set the median value to the output array
                if channels == 1:
                    result_arr[i, j] = median_value
                else:
                    result_arr[i, j, c] = median_value

    return result_arr


# [[ 1, 2, 3],
#  [ 4, 5, 6],
#  [ 7, 8, 9]]
#
# [[ 0, 0, 0, 0, 0],
#  [ 0, 1, 2, 3, 0],
#  [ 0, 4, 5, 6, 0],
#  [ 0, 7, 8, 9, 0],
#  [ 0, 0, 0, 0, 0]]
#
# [[ 0, 0, 0],
#  [ 0, 1, 2],
#  [ 0, 4, 5]]
#
# [0, 0, 0, 0, 0, 1, 2, 4, 5] -> 0


def doGeometricMeanFilter(param, arr):
    print("Function doGeometricMeanFilter invoked with param: " + param)
    return arr
