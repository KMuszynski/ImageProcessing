import numpy as np


def doMedianFilter(param, arr):
    print("Function doMedianFilter invoked with param: " + str(param))

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
    result_arr = np.zeros_like(arr)

    for c in range(channels):
        if channels == 1:
            padded_arr = np.pad(arr, pad_size, mode='reflect')
        else:
            padded_arr = np.pad(arr[:, :, c], pad_size, mode='reflect')

        # Apply median filter
        for i in range(height):
            for j in range(width):
                # Extract the neighborhood window
                window = padded_arr[i:i + param, j:j + param].flatten()

                # Sort the window to find the median
                window_sorted = sorted(window)
                window_len = len(window_sorted)

                # Calculate the median manually
                if window_len % 2 == 1:
                    median_value = window_sorted[window_len // 2]  # Middle element for odd length
                else:
                    median_value = (window_sorted[window_len // 2 - 1] + window_sorted[
                        window_len // 2]) / 2  # Average of two middle elements

                # Set the median value to the output array
                if channels == 1:
                    result_arr[i, j] = median_value
                else:
                    result_arr[i, j, c] = median_value

    return result_arr


def doGeometricMeanFilter(param, arr):
    print("Function doGeometricMeanFilter invoked with param: " + str(param))

    param = int(param)
    arr = np.array(arr)

    if arr.ndim == 2:  # Grayscale
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:  # Color
        height, width, channels = arr.shape
    else:
        raise ValueError("Unexpected array shape. Please check the image format.")

    # Padding size for the kernel (param x param window)
    pad_size = param // 2
    result_arr = np.zeros_like(arr)

    for c in range(channels):
        if channels == 1:
            padded_arr = np.pad(arr, pad_size, mode='reflect')
        else:
            padded_arr = np.pad(arr[:, :, c], pad_size, mode='reflect')

        # Apply geometric mean filter
        for i in range(height):
            for j in range(width):
                # Extract the neighborhood window
                window = padded_arr[i:i + param, j:j + param].flatten()

                # Calculate the geometric mean manually
                product = 1.0
                for value in window:
                    product *= value + 1e-10  # Adding a small value to avoid log(0)

                geometric_mean = product ** (1 / len(window))

                # Set the geometric mean value to the output array
                if channels == 1:
                    result_arr[i, j] = geometric_mean
                else:
                    result_arr[i, j, c] = geometric_mean

    return result_arr
