import numpy as np


def doHorizontalFlip(arr):
    # Determine dimensions manually
    if arr.ndim == 2:  # Grayscale image (2D array)
        height = len(arr)  # Number of rows
        width = len(arr[0])  # Number of columns
    elif arr.ndim == 3:  # Color image (3D array: height x width x channels)
        height = len(arr)  # Number of rows
        width = len(arr[0])  # Number of columns
    else:
        raise ValueError("Unexpected array shape. Please check the image format.")

    print(f"Image dimensions: {width}x{height}")

    # Flip horizontally by reversing each row
    arr2 = np.copy(arr)  # Create a copy of the array to avoid modifying the original
    for i in range(height):
        arr2[i] = arr[i][::-1]  # Reverse the row to flip horizontally

    return arr2


def doVerticalFlip(arr):
    if arr.ndim not in [2, 3]:
        raise ValueError("Unexpected array shape. Please check the image format.")

    width = arr.shape[1]
    height = arr.shape[0]
    print(f"Image dimensions: {width}x{height}")

    arr2 = arr[::-1]
    return arr2


def doDiagonalFlip(arr):
    arr = doVerticalFlip(arr)
    arr = doHorizontalFlip(arr)
    return arr


def doShrink(param, arr):
    print("Function doShrink invoked with param: " + str(param))
    param = int(param)
    # Ensure input is a numpy array
    arr = np.array(arr)

    # Get original dimensions
    height, width = arr.shape[:2]  # Assumes arr is either (H, W) for grayscale or (H, W, C) for color

    print(f"Original size: {height} x {width}")

    # Calculate new dimensions
    resultHeight = int(height * (param / 100))
    resultWidth = int(width * (param / 100))

    print(f"New size: {resultHeight} x {resultWidth}")

    # Create an empty array for the resized image (preserves channels if they exist)
    if arr.ndim == 3:  # Color image (e.g., RGB)
        newArr = np.zeros((resultHeight, resultWidth, arr.shape[2]), dtype=arr.dtype)
    else:  # Grayscale image
        newArr = np.zeros((resultHeight, resultWidth), dtype=arr.dtype)

    # Perform nearest-neighbor interpolation
    for i in range(resultHeight):
        for j in range(resultWidth):
            # Map to the nearest pixel in the original image
            orig_i = int(i * (height / resultHeight))
            orig_j = int(j * (width / resultWidth))

            # Copy the pixel from the original array to the new array
            newArr[i, j] = arr[orig_i, orig_j]

    return newArr


def doEnlarge(param, arr):
    print("Function doEnlarge invoked with param: " + param)

    param = int(param)
    # Ensure input is a numpy array
    arr = np.array(arr)

    # Get original dimensions
    height, width = arr.shape[:2]  # Assumes arr is either (H, W) for grayscale or (H, W, C) for color

    print(f"Original size: {height} x {width}")

    # Calculate new dimensions
    resultHeight = int(height * (param))
    resultWidth = int(width * (param))

    print(f"New size: {resultHeight} x {resultWidth}")

    # Create an empty array for the resized image (preserves channels if they exist)
    if arr.ndim == 3:  # Color image (e.g., RGB)
        newArr = np.zeros((resultHeight, resultWidth, arr.shape[2]), dtype=arr.dtype)
    else:  # Grayscale image
        newArr = np.zeros((resultHeight, resultWidth), dtype=arr.dtype)

    for i in range(resultHeight):
        for j in range(resultWidth):
            # Map to the nearest pixel in the original image
            orig_i = int(i * (height / resultHeight))
            orig_j = int(j * (width / resultWidth))

            # Copy the pixel from the original array to the new array
            newArr[i, j] = arr[orig_i, orig_j]

    return newArr  # Ensure it returns arr, even if no changes are made
