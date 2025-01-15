from PIL import Image
import numpy as np


def create_binary_bmp_from_points(points, size=(8, 6)):
    """
    Creates a binary BMP image from a set of points.

    Args:
        points (set of tuples): A set of (x, y) points to set to 1 in the image.
        size (tuple): The size of the image (width, height).
    """
    array = np.zeros(size, dtype=np.uint8)
    for x, y in points:
        array[y, x] = 1  # Note: (x, y) corresponds to (col, row) in the array
    array *= 255  # Convert binary to 0 and 255 for BMP
    img = Image.fromarray(array, mode="L")
    img.save("binary_image.bmp", format="BMP")
    print("Binary image saved as 'binary_image.bmp'.")


def create_binary_bmp_from_array(array):
    """
    Creates a binary BMP image from a NumPy array.

    Args:
        array (np.ndarray): A 2D array with binary values (0 or 1).
    """
    binary_array = (array * 255).astype(np.uint8)  # Convert binary to 0 and 255 for BMP
    img = Image.fromarray(binary_array, mode="L")
    img.save("binary_image_from_array.bmp", format="BMP")
    print("Binary image saved as 'binary_image_from_array.bmp'.")


# Example usage with points
# points = {(2, 1), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4)}
# create_binary_bmp_from_points(points, size=(8, 6))

# Example usage with NumPy array
# array = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
# ], dtype=int)

array = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# array = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
# ], dtype=int)

# array = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 1, 0],
#     [0, 0, 1, 1, 0, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 0],
#     [0, 0, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=int)

# array = np.array([
#     [0, 0, 0, 0, 1, 0, 0, 0, 1],
#     [0, 1, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=int)

# array = np.array([
#     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=int)

create_binary_bmp_from_array(array)
