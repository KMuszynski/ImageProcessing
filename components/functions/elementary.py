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
    param = int(param)
    if param < -100 or param > 100:
        print("error: contrast value must be between -100 and 100")
        exit()
    try:
        contrast_factor = 1 + (float(param) / 100)
    except ValueError:
        print("error: contrast value must be a number")
        exit()

    pivot = 128  # Midpoint for 8-bit images
    arr = arr.astype(np.int16)  # Prevent overflow
    new_arr = pivot + contrast_factor * (arr - pivot)
    new_arr = np.clip(new_arr, 0, 255)
    print("Function doContrast invoked with param: " + str(param))
    return new_arr.astype(np.uint8)


def doNegative(arr):
    arr = 255 - arr
    return arr
