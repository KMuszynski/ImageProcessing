import numpy as np


def generate_histogram_image(image_array, channel=None, output_path="output_histogram.bmp"):
    """
    Calculate the histogram of the specified channel, render it as an image, and save it to disk.

    Parameters:
        image_array (2D or 3D list or numpy array): The input image data.
        channel (int or None): The channel to process (0=R, 1=G, 2=B, None for grayscale).
        output_path (str): Path to save the histogram image.

    Returns:
        None
    """
    # Extract the channel or grayscale data
    if len(image_array.shape) == 3:  # Color image
        if channel is None:
            raise ValueError("Channel must be specified for color images.")
        channel_data = image_array[:, :, channel]
    elif len(image_array.shape) == 2:  # Grayscale image
        channel_data = image_array
    else:
        raise ValueError("Invalid image array shape!")

    # Calculate histogram
    histogram = [0] * 256
    for row in channel_data:
        for pixel in row:
            histogram[pixel] += 1

    # Normalize histogram for visualization
    height, width = 256, 256
    max_freq = max(histogram)
    normalized_histogram = [int((value / max_freq) * height) for value in histogram]

    # Render as image
    histogram_image = np.zeros((height, width), dtype=np.uint8)
    for x, freq in enumerate(normalized_histogram):
        for y in range(height - freq, height):  # Draw vertical bar
            histogram_image[y, x] = 255  # White bar

    # Save the histogram image as BMP with a grayscale palette
    with open(output_path, "wb") as f:
        # BMP header
        f.write(b"BM")
        file_size = 54 + 1024 + (width * height)  # Header size + color palette size + pixel data size
        f.write(file_size.to_bytes(4, byteorder="little"))
        f.write((0).to_bytes(4, byteorder="little"))  # Reserved
        f.write((54 + 1024).to_bytes(4, byteorder="little"))  # Pixel data offset
        f.write((40).to_bytes(4, byteorder="little"))  # Info header size
        f.write(width.to_bytes(4, byteorder="little"))
        f.write(height.to_bytes(4, byteorder="little"))
        f.write((1).to_bytes(2, byteorder="little"))  # Planes
        f.write((8).to_bytes(2, byteorder="little"))  # Bits per pixel
        f.write((0).to_bytes(4, byteorder="little"))  # Compression
        f.write((width * height).to_bytes(4, byteorder="little"))  # Image size
        f.write((2835).to_bytes(4, byteorder="little"))  # X pixels per meter
        f.write((2835).to_bytes(4, byteorder="little"))  # Y pixels per meter
        f.write((256).to_bytes(4, byteorder="little"))  # Total colors
        f.write((0).to_bytes(4, byteorder="little"))  # Important colors

        # Color palette (grayscale)
        for i in range(256):
            f.write(bytes([i, i, i, 0]))

        # Pixel data
        for row in histogram_image[::-1]:  # BMP stores rows bottom-to-top
            f.write(row)

    print(f"Histogram image saved to {output_path}.")
