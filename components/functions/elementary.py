import numpy as np


def doBrightness(param, arr):
    try:
        brightness_value = int(param)
    except ValueError:
        print("error: brightness value must be an integer")
        exit()

    if brightness_value < -100 or brightness_value > 100:
        print("error: brightness must be between -100 and 100")
        exit()

    print("Function doBrightness invoked with param: " + str(param))
    arr = arr.astype(np.int16)
    arr = arr + brightness_value
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


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
