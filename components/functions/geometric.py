# components/functions/geometric.py

import numpy as np

def doHorizontalFlip(arr):
    # Determine dimensions manually
    if arr.ndim == 2:  # Grayscale image
        height = len(arr)        # Number of rows
        width = len(arr[0])     # Number of columns
    elif arr.ndim == 3:  # Color image (e.g., RGB)
        height = len(arr)        # Number of rows
        width = len(arr[0])     # Number of columns
    else:
        raise ValueError("Unexpected array shape. Please check the image format.")

    print(f"Image dimensions: {width}x{height}")

    arr2 = arr
    for i in arr:
        arr2[len(arr2)-i] = arr[i]
    return arr2

def doVerticalFlip(param, arr):
    print("Function doVerticalFlip invoked with param: " + param)
    # Placeholder for future implementation
    return arr  # Ensure it returns arr, even if no changes are made

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
