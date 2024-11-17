import numpy as np


def apply_rayleigh_pdf(image_array, sigma=50):
    """
    Enhance image quality using the Rayleigh probability density function.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        sigma (float): Scale parameter for the Rayleigh PDF.

    Returns:
        numpy.ndarray: Enhanced image array.
    """
    # If the image is grayscale, process it directly
    if len(image_array.shape) == 2:
        return _apply_rayleigh_to_channel(image_array, sigma)

    # If the image is RGB, process each channel independently
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        enhanced_channels = []
        for channel in range(3):  # Loop over R, G, B channels
            enhanced_channel = _apply_rayleigh_to_channel(image_array[:, :, channel], sigma)
            enhanced_channels.append(enhanced_channel)

        # Combine the processed channels back into an RGB image
        return np.stack(enhanced_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format. Must be grayscale or RGB.")


def _apply_rayleigh_to_channel(channel_array, sigma):
    """
    Apply the Rayleigh PDF to a single channel.

    Parameters:
        channel_array (numpy.ndarray): Single-channel image array.
        sigma (float): Scale parameter for the Rayleigh PDF.

    Returns:
        numpy.ndarray: Enhanced single-channel array.
    """
    # Normalize the channel to range [0, 1]
    normalized_channel = channel_array / 255.0

    # Apply the Rayleigh PDF
    enhanced_channel = (normalized_channel / (sigma ** 2)) * np.exp(-normalized_channel ** 2 / (2 * sigma ** 2))

    # Normalize the result to range [0, 255]
    enhanced_channel = enhanced_channel / np.max(enhanced_channel)  # Ensure the max value is 1
    return (enhanced_channel * 255).astype(np.uint8)
