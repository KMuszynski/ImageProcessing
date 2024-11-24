from PIL import Image
import numpy as np
from components.functions.similarity import doMeanSquareError, doPeakMeanSquareError, doSignalToNoiseRatio, \
    doPeakSignalToNoiseRatio, doMaximumDifference

# Image without noise
original_image = Image.open("components/images/lenac.bmp")
original_arr = np.array(original_image.getdata())

noisy_image = Image.open("./components/images/noise-color/lenac_normal3.bmp")
noisy_arr = np.array(noisy_image.getdata())

# Noise removal result
result_image = Image.open("./result.bmp")
result_arr = np.array(result_image.getdata())


def compareOriginalNoisy():
    mse = doMeanSquareError(original_arr, noisy_arr)
    pmse = doPeakMeanSquareError(original_arr, noisy_arr)
    snr = doSignalToNoiseRatio(original_arr, noisy_arr)
    psnr = doPeakSignalToNoiseRatio(original_arr, noisy_arr)
    md = doMaximumDifference(original_arr, noisy_arr)

    print(f"Mean Square Error (MSE): {mse}")
    print(f"Peak Mean Square Error (PMSE): {pmse}")
    print(f"Signal to Noise Ratio (SNR): {snr}")
    print(f"Peak Signal to Noise Ratio (PSNR): {psnr}")
    print(f"Maximum Difference (MD): {md}")


def compareOriginalResult():
    mse = doMeanSquareError(original_arr, result_arr)
    pmse = doPeakMeanSquareError(original_arr, result_arr)
    snr = doSignalToNoiseRatio(original_arr, result_arr)
    psnr = doPeakSignalToNoiseRatio(original_arr, result_arr)
    md = doMaximumDifference(original_arr, result_arr)

    print(f"Mean Square Error (MSE): {mse}")
    print(f"Peak Mean Square Error (PMSE): {pmse}")
    print(f"Signal to Noise Ratio (SNR): {snr}")
    print(f"Peak Signal to Noise Ratio (PSNR): {psnr}")
    print(f"Maximum Difference (MD): {md}")


compareOriginalNoisy()
print("\n")
compareOriginalResult()
