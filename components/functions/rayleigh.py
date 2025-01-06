import numpy as np
from PIL import ImageOps, Image

from components.functions.histogram import calculate_manual_histogram


def apply_rayleigh_pdf_histogram(image_array, g_min=0, g_max=255, q=0.999):
    """
    Enhance an image using a histogram-based Rayleigh final probability density function.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        g_min (int/float): Desired minimum brightness in the output.
        g_max (int/float): Desired maximum brightness in the output.
        q (float): A quantile close to 1 used to calculate alpha.

    Returns:
        numpy.ndarray: Enhanced image array.
    """

    def _calculate_alpha(g_min, g_max, q):
        return (g_max - g_min) / np.sqrt(-2 * np.log(1 - q))

    def _apply_rayleigh_to_channel(channel_array, histogram, alpha, g_min):
        """
        Enhance a single channel using a histogram and Rayleigh PDF.
        """
        cdf = np.cumsum(histogram) / np.sum(histogram)  # CDF
        enhanced_channel = np.zeros_like(channel_array, dtype=float)

        for f in range(256):  # For each intensity
            cdf_clamped = np.clip(cdf[f], 1e-8, 1 - 1e-8)  # Clamp CDF to avoid log(0)
            enhanced_channel[channel_array == f] = g_min + alpha * np.sqrt(-2 * np.log(1 - cdf_clamped))

        return np.clip(enhanced_channel, 0, 255).astype(np.uint8)

    alpha = _calculate_alpha(g_min, g_max, q)

    if len(image_array.shape) == 2:  # Grayscale
        histogram = calculate_manual_histogram(image_array)
        return _apply_rayleigh_to_channel(image_array, histogram, alpha, g_min)

    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
        enhanced_channels = []
        for channel_index in range(3):  # Process channels separately
            histogram = calculate_manual_histogram(image_array, channel=channel_index)
            enhanced_channel = _apply_rayleigh_to_channel(image_array[:, :, channel_index], histogram, alpha, g_min)
            enhanced_channels.append(enhanced_channel)

        return np.stack(enhanced_channels, axis=-1)

    else:
        raise ValueError("Unsupported image format. Must be grayscale or RGB.")
