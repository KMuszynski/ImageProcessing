import numpy as np
from components.functions.histogram import calculate_manual_histogram


def apply_rayleigh_pdf_histogram(image_array, alpha=50):
    """
    Enhance image quality using a histogram-based Rayleigh PDF method.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        alpha (float): Scale parameter for the Rayleigh PDF.

    Returns:
        numpy.ndarray: Enhanced image array.
    """
    # If the image is grayscale
    if len(image_array.shape) == 2:
        histogram = calculate_manual_histogram(image_array)
        return _enhance_with_histogram(image_array, histogram, alpha)

    # If the image is RGB
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Flatten all channels to calculate a global histogram
        combined_histogram = calculate_manual_histogram(image_array.flatten())
        enhanced_channels = []
        for channel in range(3):  # Process R, G, B channels together
            enhanced_channel = _enhance_with_histogram(
                image_array[:, :, channel], combined_histogram, alpha
            )
            enhanced_channels.append(enhanced_channel)

        # Combine the enhanced channels back into an RGB image
        return np.stack(enhanced_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format. Must be grayscale or RGB.")


def _enhance_with_histogram(channel_array, histogram, alpha):
    """
    Enhance a single channel using a histogram and Rayleigh PDF.

    Parameters:
        channel_array (numpy.ndarray): Single-channel image array.
        histogram (numpy.ndarray): Precomputed histogram (256 bins).
        alpha (float): Scale parameter for the Rayleigh PDF.

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
        if cdf[f] > 0:  # Avoid log(0)
            enhanced_channel[channel_array == f] = (
                alpha * np.sqrt(2 * np.log(1 / cdf[f]))
            )

    # Scale the result to [0, 255]
    enhanced_channel = np.clip(enhanced_channel, 0, 255)
    return enhanced_channel.astype(np.uint8)
