import numpy as np
from PIL import Image
import sys
import os
import time

# Load the image
image_path = "./components/images/c_lenac_small.bmp"
image = Image.open(image_path)

# If the image is not RGB, convert it to RGB ('RGB' mode)
if image.mode != "RGB":
    image = image.convert("RGB")

# Convert the image to a numpy array (RGB)
arr = np.array(image)

# Parameters
height, width, _ = arr.shape  # Image dimensions

# Function to decimate (downsample) the image in the spatial domain by a factor of 2
def decimate_image(image, factor=2):
    return image[::factor, ::factor]

# 1D Fourier Transform function (for rows or columns) with decimation
def dft_1d_decimated(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the DFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(-2j * np.pi * n * k / N)
        result[n] = sum
    return result

# 2D Fourier Transform function (applies DFT to rows and columns) with decimation
def dft_2d_decimated(image):
    height, width = image.shape
    # Decimate the image first
    image_decimated = decimate_image(image)
    height, width = image_decimated.shape  # New dimensions after decimation

    # Apply the 1D DFT to each row
    row_transform = np.array([dft_1d_decimated(image_decimated[row]) for row in range(height)])
    # Apply the 1D DFT to each column of the result
    col_transform = np.array([dft_1d_decimated(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

# 1D Fourier Transform function (for rows or columns) without decimation (slow)
def dft_1d(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the DFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(-2j * np.pi * n * k / N)
        result[n] = sum
    return result

# 2D Fourier Transform function (applies DFT to rows and columns) without decimation (slow)
def dft_2d(image):
    height, width = image.shape
    # Apply the 1D DFT to each row
    row_transform = np.array([dft_1d(image[row]) for row in range(height)])
    # Apply the 1D DFT to each column of the result
    col_transform = np.array([dft_1d(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

# 1D Inverse Fourier Transform function (for rows or columns) with decimation
def idft_1d_decimated(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the IDFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(2j * np.pi * n * k / N)
        result[n] = sum / N
    return result

# 2D Inverse Fourier Transform function with decimation
def idft_2d_decimated(f_transform):
    height, width = f_transform.shape
    # Apply the 1D IDFT to each row
    row_transform = np.array([idft_1d_decimated(f_transform[row]) for row in range(height)])
    # Apply the 1D IDFT to each column of the result
    col_transform = np.array([idft_1d_decimated(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

# 1D Inverse Fourier Transform function (for rows or columns) without decimation (slow)
def idft_1d(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the IDFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(2j * np.pi * n * k / N)
        result[n] = sum / N
    return result

# 2D Inverse Fourier Transform function without decimation (slow)
def idft_2d(f_transform):
    height, width = f_transform.shape
    # Apply the 1D IDFT to each row
    row_transform = np.array([idft_1d(f_transform[row]) for row in range(height)])
    # Apply the 1D IDFT to each column of the result
    col_transform = np.array([idft_1d(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

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

# Save Fourier transform of each channel
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

# Check if --fast argument is passed
if "--fast" in sys.argv:
    start_time = time.time()

    # Apply the 2D Fourier Transform to each RGB channel with decimation
    f_transform_r = dft_2d_decimated(arr[:, :, 0])  # Red channel
    f_transform_g = dft_2d_decimated(arr[:, :, 1])  # Green channel
    f_transform_b = dft_2d_decimated(arr[:, :, 2])  # Blue channel

    # Apply the 2D Inverse Fourier Transform to each channel to get the reconstructed channels
    reconstructed_r = idft_2d_decimated(f_transform_r)
    reconstructed_g = idft_2d_decimated(f_transform_g)
    reconstructed_b = idft_2d_decimated(f_transform_b)

    # Take the real part of the reconstructed images (imaginary parts should be negligible)
    reconstructed_r_real = np.real(reconstructed_r)
    reconstructed_g_real = np.real(reconstructed_g)
    reconstructed_b_real = np.real(reconstructed_b)

    # Reconstruct the image by stacking the RGB channels back together
    reconstructed_image_rgb = np.stack((reconstructed_r_real, reconstructed_g_real, reconstructed_b_real), axis=-1)

    # Clip values to ensure they are in the valid range [0, 255] for uint8
    reconstructed_image_rgb = np.clip(reconstructed_image_rgb, 0, 255).astype(np.uint8)

    # Save the reconstructed image directly using Pillow
    reconstructed_im = Image.fromarray(reconstructed_image_rgb)
    reconstructed_im.save("reconstructed-image.bmp")

    end_time = time.time()
    print("Fast mode: Reconstruction complete.")
    print(f"Time taken (Fast): {end_time - start_time:.4f} seconds")

# Save the original image as BMP
original_im = Image.fromarray(arr.astype(np.uint8))
original_im.save("original-image.bmp")

print("Original image saved as 'original-image.bmp'")

# Save Fourier transforms for each channel (if --fast argument was passed)
if "--fast" in sys.argv:
    # Save Fourier transforms for each channel
    save_fourier_transform(f_transform_r, 'red')
    save_fourier_transform(f_transform_g, 'green')
    save_fourier_transform(f_transform_b, 'blue')

    print("Fast mode: Fourier transforms for each channel saved.")

# Slow method (no decimation, same as before)
if "--slow" in sys.argv:
    start_time = time.time()

    # Step 1: Apply the 2D Fourier Transform to each RGB channel
    f_transform_r = dft_2d(arr[:, :, 0])  # Red channel
    f_transform_g = dft_2d(arr[:, :, 1])  # Green channel
    f_transform_b = dft_2d(arr[:, :, 2])  # Blue channel

    # Step 2: Apply the 2D Inverse Fourier Transform to each channel to get the reconstructed channels
    reconstructed_r = idft_2d(f_transform_r)
    reconstructed_g = idft_2d(f_transform_g)
    reconstructed_b = idft_2d(f_transform_b)

    # Take the real part of the reconstructed images (imaginary parts should be negligible)
    reconstructed_r_real = np.real(reconstructed_r)
    reconstructed_g_real = np.real(reconstructed_g)
    reconstructed_b_real = np.real(reconstructed_b)

    # Reconstruct the image by stacking the RGB channels back together
    reconstructed_image_rgb = np.stack((reconstructed_r_real, reconstructed_g_real, reconstructed_b_real), axis=-1)

    # Clip values to ensure they are in the valid range [0, 255] for uint8
    reconstructed_image_rgb = np.clip(reconstructed_image_rgb, 0, 255).astype(np.uint8)

    # Save the reconstructed image directly using Pillow
    reconstructed_im = Image.fromarray(reconstructed_image_rgb)
    reconstructed_im.save("reconstructed-image.bmp")

    end_time = time.time()
    print("Slow mode: Reconstruction complete.")
    print(f"Time taken (Slow): {end_time - start_time:.4f} seconds")

    # Save Fourier transforms for each channel (slow mode)
    save_fourier_transform(f_transform_r, 'red')
    save_fourier_transform(f_transform_g, 'green')
    save_fourier_transform(f_transform_b, 'blue')

    print("Slow mode: Fourier transforms for each channel saved.")
