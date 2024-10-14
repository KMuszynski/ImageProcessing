# components/functions/geometric.py
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
    print("Vertical flip applied successfully!")
    return arr2


def doDiagonalFlip(param, arr):
    print("Function doDiagonalFlip invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made


def doShrink(param, arr):
    print("Function doShrink invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made


def doEnlarge(param, arr):
    print("Function doEnlarge invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made
