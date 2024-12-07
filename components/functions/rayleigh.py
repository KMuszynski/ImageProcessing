import numpy as np
from PIL import ImageOps, Image

from components.functions.histogram import calculate_manual_histogram


def apply_rayleigh_pdf_histogram(image_array, g_min=0, g_max=255, q=0.999):  # alpha needs to be calculated from the gmean and gmax
    """
    Enhance image quality using a histogram-based Rayleigh final probability density function.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        g_min (int/float): Desired minimum brightness in the output.
        g_max (int/float): Desired maximum brightness in the output.
        q (float): A quantile close to 1 used to calculate alpha.

    Returns:
        numpy.ndarray: Enhanced image array.
    """

    # Calculate alpha based on g_min, g_max, and q
    alpha = (g_max - g_min) / np.sqrt(-2 * np.log(1 - q))

    # If the image is grayscale
    if len(image_array.shape) == 2:
        histogram = calculate_manual_histogram(image_array)
        return _enhance_with_histogram(image_array, histogram, alpha, g_min)

    # If the image is RGB
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert to grayscale for global histogram computation
        grayscale_image = ImageOps.grayscale(Image.fromarray(image_array))
        grayscale_array = np.array(grayscale_image)
        histogram = calculate_manual_histogram(grayscale_array)

        # Apply the same transformation to all channels
        enhanced_channels = []
        for channel in range(3):  # Process R, G, B channels together
            enhanced_channel = _enhance_with_histogram(
                image_array[:, :, channel], histogram, alpha, g_min
            )
            enhanced_channels.append(enhanced_channel)

        # Combine the enhanced channels back into an RGB image
        return np.stack(enhanced_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format. Must be grayscale or RGB.")


def _enhance_with_histogram(channel_array, histogram, alpha, g_min):
    """
    Enhance a single channel using a histogram and Rayleigh PDF.

    Parameters:
        channel_array (numpy.ndarray): Single-channel image array.
        histogram (numpy.ndarray): Precomputed histogram (256 bins).
        alpha (float): Scale parameter for the Rayleigh PDF.
        g_min (int/float): Minimum output brightness.

    Returns:
        numpy.ndarray: Enhanced single-channel array.
    """

    # Normalize pixel values to range [0, 1]
    normalized_channel = channel_array / 255.0

    # Use histogram to calculate cumulative distribution function (CDF)
    cdf = np.cumsum(histogram) / np.sum(histogram)

    # Apply Rayleigh PDF formula
    enhanced_channel = np.zeros_like(normalized_channel, dtype=float)
    for f in range(256):
        # Clamp to avoid log(0)
        cdf_clamped = np.clip(cdf[f], 1e-8, 1 - 1e-8)
        # Use the correct inverse transform:
        # g(f) = g_min + alpha * sqrt(-2 ln(1 - cdf_clamped))
        enhanced_channel[channel_array == f] = g_min + alpha * np.sqrt(-2 * np.log(1 - cdf_clamped))

        # Normalize to range [0, 255]
    enhanced_channel = np.clip(enhanced_channel, 0, 255)
    return enhanced_channel.astype(np.uint8)
