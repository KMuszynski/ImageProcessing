from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def calculate_manual_histogram(image_array, channel=None):
    """
    Calculate a histogram for a grayscale or single channel of an RGB image.

    Parameters:
        image_array (numpy.ndarray): Image array (grayscale or RGB).
        channel (int or None): Index of the channel to process (0: Red, 1: Green, 2: Blue).
                               If None, processes grayscale.

    Returns:
        numpy.ndarray: Histogram array with 256 bins.
    """
    histogram = np.zeros(256, dtype=int)

    if channel is None:
        # Grayscale image
        for pixel in image_array.flatten():
            histogram[pixel] += 1
    else:
        # Single channel of RGB image
        for pixel in image_array[:, :, channel].flatten():
            histogram[pixel] += 1

    return histogram


def calculate_save_histogram(image_path, channel="gray", save_path="histogram.png"):
    """
    Calculate and save the histogram of the specified channel from the given image.

    Parameters:
        image_path (str): Path to the input image.
        channel (str): Channel to process ('gray', 'red', 'green', 'blue').
        save_path (str): Path to save the histogram image.
    """
    image = Image.open(image_path)

    if channel == "gray":
        # Convert to grayscale if required
        # gray_image = image.convert("L")
        pixel_values = np.array(image)
        histogram = calculate_manual_histogram(pixel_values)

        # Plot the histogram for grayscale
        plt.bar(range(256), histogram, color='gray', width=1.0)
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    else:
        # Ensure the image is in RGB mode
        # rgb_image = image.convert("RGB")
        pixel_values = np.array(image)

        # Extract the desired channel
        channel_map = {"red": 0, "green": 1, "blue": 2}
        if channel not in channel_map:
            raise ValueError(f"Invalid channel '{channel}'. Choose from 'red', 'green', 'blue', or 'gray'.")

        histogram = calculate_manual_histogram(pixel_values, channel_map[channel])

        # Plot the histogram for the specified channel
        plt.bar(range(256), histogram, color=channel, width=1.0)
        plt.title(f"{channel.capitalize()} Channel Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    # Save the histogram as an image
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved as {save_path}")
