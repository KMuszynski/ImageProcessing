import numpy as np


def ll_operator(arr):
    """
    Apply the LL operator to a grayscale image.
    g(n,m) = (1/4) * |log((x(n,m)^4) / (A1 * A3 * A5 * A7))|
    """
    epsilon = 1e-6  # Small constant to avoid log(0) or division by zero
    if arr.ndim == 3:  # Convert RGB to grayscale
        arr = np.mean(arr, axis=2).astype(np.float32)

    padded_arr = np.pad(arr, pad_width=1, mode='reflect')  # Reflective padding
    output = np.zeros_like(arr, dtype=np.float32)

    for i in range(1, padded_arr.shape[0] - 1):
        for j in range(1, padded_arr.shape[1] - 1):
            center_pixel = max(padded_arr[i, j], epsilon)  # Avoid zero center pixel

            # Immediate neighbors (up, right, down, left)
            A1 = max(padded_arr[i - 1, j], epsilon)  # Up
            A3 = max(padded_arr[i, j + 1], epsilon)  # Right
            A5 = max(padded_arr[i + 1, j], epsilon)  # Down
            A7 = max(padded_arr[i, j - 1], epsilon)  # Left

            # Calculate the LL operator result
            neighbor_product = A1 * A3 * A5 * A7
            result = (1 / 4) * np.abs(np.log((center_pixel ** 4) / neighbor_product))

            output[i - 1, j - 1] = result  # Place result in the output array

    # Normalize the output to the range [0, 255]
    output -= output.min()  # Shift the minimum to 0
    output = (output / output.max()) * 255  # Scale to [0, 255]

    return output.astype(np.uint8)
