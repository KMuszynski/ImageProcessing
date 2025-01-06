import numpy as np

# 1D Fourier Transform function (for rows or columns)
def dft_1d(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the DFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(-2j * np.pi * n * k / N)
        result[n] = sum
    return result

# 2D Fourier Transform function (applies DFT to rows and columns)
def dft_2d(image):
    height, width = image.shape
    # Apply the 1D DFT to each row
    row_transform = np.array([dft_1d(image[row]) for row in range(height)])
    # Apply the 1D DFT to each column of the result
    col_transform = np.array([dft_1d(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

# 1D Inverse Fourier Transform function (for rows or columns)
def idft_1d(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)  # Result of the IDFT
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += signal[k] * np.exp(2j * np.pi * n * k / N)
        result[n] = sum / N
    return result

# 2D Inverse Fourier Transform function
def idft_2d(f_transform):
    height, width = f_transform.shape
    # Apply the 1D IDFT to each row
    row_transform = np.array([idft_1d(f_transform[row]) for row in range(height)])
    # Apply the 1D IDFT to each column of the result
    col_transform = np.array([idft_1d(row_transform[:, col]) for col in range(width)])
    return col_transform.T  # Transpose back to original dimensions

# Function to apply Fourier Transform (2D DFT) on an image
def apply_fourier_transform(image_arr):
    # Separate the image into R, G, B channels
    f_transform_r = dft_2d(image_arr[:, :, 0])  # Red channel
    f_transform_g = dft_2d(image_arr[:, :, 1])  # Green channel
    f_transform_b = dft_2d(image_arr[:, :, 2])  # Blue channel
    return f_transform_r, f_transform_g, f_transform_b

# Function to apply Inverse Fourier Transform (2D IDFT) on transformed data
def apply_inverse_fourier_transform(f_transform_r, f_transform_g, f_transform_b):
    # Apply the 2D Inverse Fourier Transform to each channel
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
    
    return reconstructed_image_rgb
