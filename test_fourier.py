import numpy as np
from PIL import Image
import sys
import os
import time
import matplotlib.pyplot as plt

image_path = "./components/images/lenac.bmp"
image = Image.open(image_path)

if image.mode != "RGB":
    image = image.convert("RGB")

arr = np.array(image)

# Parameters
height, width, _ = arr.shape # Ignore 3


###############################################################################
# 2) FAST FFT AND IFFT (1D, COOLEY–TUKEY)
###############################################################################
def fft_1d(signal):
    """
    1D Cooley–Tukey Fast Fourier Transform (recursive).
    Assumes len(signal) is a power of two.
    """
    N = len(signal)
    if N <= 1:
        return signal
    # Splitting into even and odd indices
    even = fft_1d(signal[0::2])
    odd = fft_1d(signal[1::2])

    # Combining
    combined = np.zeros(N, dtype=complex)
    half = N // 2
    for k in range(half):
        # Twiddle factor
        t = np.exp(-2j * np.pi * k / N) * odd[k]
        combined[k] = even[k] + t
        combined[k + half] = even[k] - t
    return combined


def ifft_1d(signal):
    """
    1D Inverse FFT.
    Uses the property: IFFT(x) = 1/N * conj(FFT(conj(x))).
    """
    # Conjugate
    conj_signal = np.conjugate(signal)
    # Forward FFT of the conjugated signal
    transformed = fft_1d(conj_signal)
    # Conjugate again and then scale by 1/N
    N = len(signal)
    return np.conjugate(transformed) / N


###############################################################################
# 3) FAST FFT AND IFFT (1D) but with decimation in the SPATIAL domain
###############################################################################
def dft_1d_decimated(signal):
    """
    1D "FAST" DFT for decimated signals (Cooley–Tukey).
    """
    return fft_1d(signal)


def idft_1d_decimated(signal):
    """
    1D "FAST" inverse DFT for decimated signals (Cooley–Tukey).
    """
    return ifft_1d(signal)


###############################################################################
# 4) 2D FFT / IFFT WITH DECIMATION
###############################################################################
def dft_2d_decimated(image):
    """
    2D DFT using decimation in the spatial domain.
    Performs FFT on rows and then columns, without shrinking the image.
    """
    # FFT on rows
    row_transform = np.array([fft_1d(image[row]) for row in range(image.shape[0])], dtype=complex)

    # FFT on columns
    col_transform = np.array([fft_1d(row_transform[:, col]) for col in range(image.shape[1])], dtype=complex)

    # Return the transposed result to match the dimensions
    return col_transform.T


def idft_2d_decimated(f_transform):
    """
    2D inverse DFT using decimation in the spatial domain.
    Performs IFFT on rows and then columns, reconstructing the original image.
    """
    # IFFT on rows
    row_transform = np.array([ifft_1d(f_transform[row]) for row in range(f_transform.shape[0])], dtype=complex)

    # IFFT on columns
    col_transform = np.array([ifft_1d(row_transform[:, col]) for col in range(f_transform.shape[1])], dtype=complex)

    # Return the transposed result to match the dimensions
    return col_transform.T


###############################################################################
# 5) SLOW DFT / IDFT
###############################################################################
def dft_1d(signal):
    """
    1D DFT (slow, O(N^2)).
    """
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        accum = 0
        for k in range(N):
            accum += signal[k] * np.exp(-2j * np.pi * n * k / N)
        result[n] = accum
    return result


def dft_2d(image):
    """
    2D DFT (slow, O(N^2 M^2)) on rows then columns.
    """
    height, width = image.shape
    # DFT for each row
    row_transform = np.array([dft_1d(image[row]) for row in range(height)], dtype=complex)
    # DFT for each column
    col_transform = np.zeros((width, height), dtype=complex)
    for col in range(width):
        col_transform[col] = dft_1d(row_transform[:, col])
    return col_transform.T


def idft_1d(signal):
    """
    1D IDFT (slow).
    """
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        accum = 0
        for k in range(N):
            accum += signal[k] * np.exp(2j * np.pi * n * k / N)
        result[n] = accum / N
    return result


def idft_2d(f_transform):
    """
    2D IDFT (slow) on rows then columns.
    """
    height, width = f_transform.shape
    # IDFT for each row
    row_transform = np.array([idft_1d(f_transform[row]) for row in range(height)], dtype=complex)
    # IDFT for each column
    col_transform = np.zeros((width, height), dtype=complex)
    for col in range(width):
        col_transform[col] = idft_1d(row_transform[:, col])
    return col_transform.T


###############################################################################
# 6) fftshift_custom
###############################################################################
def fftshift_custom(f_transform):
    height, width = f_transform.shape
    cx, cy = height // 2, width // 2
 
    top_left = f_transform[:cx, :cy]
    top_right = f_transform[:cx, cy:]
    bottom_left = f_transform[cx:, :cy]
    bottom_right = f_transform[cx:, cy:]

    f_transform_shifted = np.zeros_like(f_transform)
    f_transform_shifted[:cx, :cy] = bottom_right
    f_transform_shifted[:cx, cy:] = bottom_left
    f_transform_shifted[cx:, :cy] = top_right
    f_transform_shifted[cx:, cy:] = top_left

    return f_transform_shifted


###############################################################################
# 7) Saving the Fourier transform as an image
###############################################################################
def save_fourier_transform(f_transform, channel_name):
    f_transform_magnitude = np.abs(f_transform)
    f_transform_shifted = fftshift_custom(f_transform_magnitude)  # Shift zero frequency to center

    # Normalize the shifted Fourier transform for better visualization
    f_transform_normalized = np.log(f_transform_shifted + 1)
    f_transform_normalized -= f_transform_normalized.min()  # Shift to positive range
    f_transform_normalized /= f_transform_normalized.max()  # Normalize to [0, 1]
    f_transform_normalized *= 255  # Scale to [0, 255]

    # Convert to uint8 and save
    fourier_im = Image.fromarray(f_transform_normalized.astype(np.uint8))
    fourier_im.save(f'{channel_name}-fourier-transform.bmp')

###############################################################################
# 8*)  Add black pixels to achive original resolution
###############################################################################
def add_black_pixels(image, original_shape):
    """
    Add black pixels every second pixel in both dimensions
    to match the original image size.

    Parameters:
        image (numpy.ndarray): Reconstructed image after decimation.
        original_shape (tuple): The shape of the original image (height, width, channels).

    Returns:
        numpy.ndarray: Image with black pixels added to maintain original size.
    """
    original_height, original_width, channels = original_shape
    # Create an array of zeros with the original dimensions
    expanded_image = np.zeros((original_height, original_width, channels), dtype=image.dtype)
    
    # Place the reconstructed image pixels at the appropriate positions
    expanded_image[::2, ::2] = image
    
    # Return the expanded image
    return expanded_image

###############################################################################
# 8) SLOW vs FAST
###############################################################################
# Save original image
original_im = Image.fromarray(arr.astype(np.uint8))
original_im.save("original-image.bmp")
print("Original image saved as 'original-image.bmp'")

if "--fast" in sys.argv:
    start_time = time.time()

    # 2D FFT with decimation in spatial domain (fast)
    f_transform_r = dft_2d_decimated(arr[:, :, 0])  # Red
    f_transform_g = dft_2d_decimated(arr[:, :, 1])  # Green
    f_transform_b = dft_2d_decimated(arr[:, :, 2])  # Blue

    # 2D IFFT (fast) for each channel
    reconstructed_r = idft_2d_decimated(f_transform_r)
    reconstructed_g = idft_2d_decimated(f_transform_g)
    reconstructed_b = idft_2d_decimated(f_transform_b)

    # Take real part
    reconstructed_r_real = np.real(reconstructed_r)
    reconstructed_g_real = np.real(reconstructed_g)
    reconstructed_b_real = np.real(reconstructed_b)

    # Stack
    reconstructed_image_rgb = np.stack((reconstructed_r_real,
                                        reconstructed_g_real,
                                        reconstructed_b_real), axis=-1)

    # Clip
    reconstructed_image_rgb = np.clip(reconstructed_image_rgb, 0, 255).astype(np.uint8)

    # Save
    reconstructed_im = Image.fromarray(reconstructed_image_rgb)
    reconstructed_im.save("reconstructed-image.bmp")

    end_time = time.time()
    print("Fast mode: Reconstruction complete.")
    print(f"Time taken (Fast): {end_time - start_time:.4f} seconds")

    # Save the Fourier transforms for each channel (visualization)
    save_fourier_transform(f_transform_r, 'red')
    save_fourier_transform(f_transform_g, 'green')
    save_fourier_transform(f_transform_b, 'blue')
    print("Fast mode: Fourier transforms for each channel saved.")

if "--slow" in sys.argv:
    start_time = time.time()

    # 2D DFT for each channel (slow)
    f_transform_r = dft_2d(arr[:, :, 0])
    f_transform_g = dft_2d(arr[:, :, 1])
    f_transform_b = dft_2d(arr[:, :, 2])

    # 2D IDFT (slow)
    reconstructed_r = idft_2d(f_transform_r)
    reconstructed_g = idft_2d(f_transform_g)
    reconstructed_b = idft_2d(f_transform_b)

    # Real part
    reconstructed_r_real = np.real(reconstructed_r)
    reconstructed_g_real = np.real(reconstructed_g)
    reconstructed_b_real = np.real(reconstructed_b)

    # Stack
    reconstructed_image_rgb = np.stack((reconstructed_r_real,
                                        reconstructed_g_real,
                                        reconstructed_b_real), axis=-1)
    # Clip
    reconstructed_image_rgb = np.clip(reconstructed_image_rgb, 0, 255).astype(np.uint8)

    # Save
    reconstructed_im = Image.fromarray(reconstructed_image_rgb)
    reconstructed_im.save("reconstructed-image.bmp")

    end_time = time.time()
    print("Slow mode: Reconstruction complete.")
    print(f"Time taken (Slow): {end_time - start_time:.4f} seconds")

    # Save Fourier transforms for each channel
    save_fourier_transform(f_transform_r, 'red')
    save_fourier_transform(f_transform_g, 'green')
    save_fourier_transform(f_transform_b, 'blue')
    print("Slow mode: Fourier transforms for each channel saved.")
