from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(image_path, channel="gray", save_path="histogram.png"):
    """
    Calculate and save the histogram of the specified channel from the given image.

    Parameters:
        image_path (str): Path to the input image.
        channel (str): Channel to process ('gray', 'red', 'green', 'blue').
        save_path (str): Path to save the histogram image.
    """
    # Load the image using PIL
    image = Image.open(image_path)

    if channel == "gray":
        # Convert to grayscale if required
        gray_image = image.convert("L")
        pixel_values = np.array(gray_image).flatten()

        # Plot the histogram for grayscale
        plt.hist(pixel_values, bins=256, range=(0, 256), color='gray')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    else:
        # Ensure the image is in RGB mode
        rgb_image = image.convert("RGB")
        pixel_values = np.array(rgb_image)

        # Extract the desired channel
        channel_map = {"red": 0, "green": 1, "blue": 2}
        if channel not in channel_map:
            raise ValueError(f"Invalid channel '{channel}'. Choose from 'red', 'green', 'blue', or 'gray'.")

        channel_values = pixel_values[:, :, channel_map[channel]].flatten()

        # Plot the histogram for the specified channel
        plt.hist(channel_values, bins=256, range=(0, 256), color=channel)
        plt.title(f"{channel.capitalize()} Channel Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    # Save the histogram as an image
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved as {save_path}")
