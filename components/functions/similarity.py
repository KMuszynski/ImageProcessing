import numpy as np


def doMeanSquareError(arr, arr_noisy):
    print("Function doMeanSquareError invoked")
    mse = np.mean((arr - arr_noisy) ** 2)
    return mse


def doPeakMeanSquareError(arr, arr_noisy):
    print("Function doPeakMeanSquareError invoked")
    max_value = np.max(arr)  # Maximum value of the original image
    pmse = np.mean((arr - arr_noisy) ** 2) / (max_value ** 2)
    return pmse


def doSignalToNoiseRatio(arr, arr_noisy):
    print("Function doSignalToNoiseRatio invoked")
    signal_power = np.sum(arr ** 2)
    noise_power = np.sum((arr - arr_noisy) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def doPeakSignalToNoiseRatio(arr, arr_noisy):
    print("Function doPeakSignalToNoiseRatio invoked")
    max_value = np.max(arr)  # Maximum value of the original image
    mse = np.mean((arr - arr_noisy) ** 2)
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr


def doMaximumDifference(arr, arr_noisy):
    print("Function doMaximumDifference invoked")
    md = np.max(np.abs(arr - arr_noisy))
    return md
