# components/functions/elementary.py

import numpy as np


def doBrightness(param, arr):
    if int(param) <= 0:
        print("error divide by zero")
        exit()
    if int(param) > 100:
        print("error brightness can't exceed 100")
        exit()
    print("Function doBrightness invoked with param: " + param)
    arr = arr * (int(param) / 100)
    return arr


def doContrast(param, arr):
    contrast_factor = float(param) / 100
    pivot = 128  # Midpoint for 8-bit images

    # Calculate the new pixel values
    new_arr = pivot + contrast_factor * (arr - pivot)

    # Manually clamp values to ensure they stay within the valid range [0, 255]
    new_arr[new_arr < 0] = 0
    new_arr[new_arr > 255] = 255

    print("Function doContrast invoked with param: " + param)
    return new_arr.astype(np.uint8)


def doNegative(arr):
    arr = 255 - arr
    return arr
