import numpy as np
from PIL import Image
import sys


def dilation(B, arr, threshold=127):
    """
    Perform dilation on a binary image using a structuring element B.
    
    Parameters:
    - B: Structuring element (binary array).
    - arr: Input binary image (as a numpy array).
    - threshold: Grayscale threshold for converting to binary (default is 127).
    
    Returns:
    - In: Dilation result (binary image after dilation).
    """
    binary_image = np.where(arr > threshold, 1, 0)  # Convert to binary
    P, Q = B.shape  # Structuring element size
    In = np.zeros_like(binary_image)  # Output image

    # Perform dilation
    for i in range(P // 2, arr.shape[0] - P // 2):
        for j in range(Q // 2, arr.shape[1] - Q // 2):
            region = binary_image[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
            if np.any(region[B == 1]):  # Check if any pixel in the region is 1 where B is 1
                In[i, j] = 1  # Set the output pixel to 1 if dilation condition met

    return In


def erosion(B, arr):
    """Erosion operation using structuring element B on image arr."""
    arr = convert_to_grayscale_and_binary(arr)
    rows, cols = arr.shape
    new_arr = np.ones_like(arr)

    for x in range(rows):
        for y in range(cols):
            neighborhood = get_neighborhood(arr, x, y, B)
            if not all(arr[nx, ny] == 1 for nx, ny in neighborhood):
                new_arr[x, y] = 0

    return new_arr.astype(np.uint8)


def opening(B, arr, threshold=127):
    """
    Perform opening operation (erosion followed by dilation) on a binary image.
    
    Parameters:
    - B: Structuring element (binary array).
    - arr: Input binary image (as a numpy array).
    - threshold: Grayscale threshold for converting to binary (default is 127).
    
    Returns:
    - Opened binary image.
    """
    # Convert the image to binary
    binary_image = np.where(arr > threshold, 1, 0)

    # Erosion: For each pixel, check if the neighborhood fits the structuring element.
    P, Q = B.shape  # Structuring element size
    eroded = np.zeros_like(binary_image)
    padded_image = np.pad(binary_image, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='constant', constant_values=0)

    # Perform erosion
    for i in range(P // 2, padded_image.shape[0] - P // 2):
        for j in range(Q // 2, padded_image.shape[1] - Q // 2):
            region = padded_image[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
            if np.all(region[B == 1] == 1):  # Check if region matches the structuring element
                eroded[i - P // 2, j - Q // 2] = 1

    # Dilation: For each pixel, check if the neighborhood fits the structuring element.
    dilated = np.zeros_like(eroded)
    padded_eroded = np.pad(eroded, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='constant', constant_values=0)

    # Perform dilation
    for i in range(P // 2, padded_eroded.shape[0] - P // 2):
        for j in range(Q // 2, padded_eroded.shape[1] - Q // 2):
            region = padded_eroded[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
            if np.any(region[B == 1] == 1):  # If any pixel in the region matches the structuring element
                dilated[i - P // 2, j - Q // 2] = 1

    return dilated.astype(np.uint8)


def closing(B, arr, threshold=127):
    """
    Perform closing operation (dilation followed by erosion) on a binary image.
    
    Parameters:
    - B: Structuring element (binary array).
    - arr: Input binary image (as a numpy array).
    - threshold: Grayscale threshold for converting to binary (default is 127).
    
    Returns:
    - Closed binary image.
    """
    # Convert the image to binary
    binary_image = np.where(arr > threshold, 1, 0)

    # Dilation: For each pixel, check if the neighborhood fits the structuring element.
    P, Q = B.shape  # Structuring element size
    dilated = np.zeros_like(binary_image)
    padded_image = np.pad(binary_image, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='constant', constant_values=0)

    # Perform dilation
    for i in range(P // 2, padded_image.shape[0] - P // 2):
        for j in range(Q // 2, padded_image.shape[1] - Q // 2):
            region = padded_image[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
            if np.any(region[B == 1] == 1):  # If any pixel in the region matches the structuring element
                dilated[i - P // 2, j - Q // 2] = 1

    # Erosion: For each pixel, check if the neighborhood fits the structuring element.
    eroded = np.zeros_like(dilated)
    padded_dilated = np.pad(dilated, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='constant', constant_values=0)

    # Perform erosion
    for i in range(P // 2, padded_dilated.shape[0] - P // 2):
        for j in range(Q // 2, padded_dilated.shape[1] - Q // 2):
            region = padded_dilated[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
            if np.all(region[B == 1] == 1):  # Check if region matches the structuring element
                eroded[i - P // 2, j - Q // 2] = 1

    return eroded.astype(np.uint8)


def hmt_transformation(B, arr):
    """Hit-or-Miss Transformation (HMT): Find pixels surrounded by B in the background."""
    arr = convert_to_grayscale_and_binary(arr)
    rows, cols = arr.shape
    new_arr = np.zeros_like(arr)

    for x in range(rows):
        for y in range(cols):
            neighborhood = get_neighborhood(arr, x, y, B)
            if all(arr[nx, ny] == 1 for nx, ny in neighborhood):
                new_arr[x, y] = 1

    return new_arr.astype(np.uint8)


def iterative_morphological_operation(B, arr, p, threshold=127):
    """
    Perform iterative morphological operation starting from a point p.
    
    Parameters:
    - B: Structuring element (binary array).
    - arr: Input binary image (as a numpy array).
    - p: Starting point (x, y) in the image.
    - threshold: Grayscale threshold for converting to binary (default is 127).
    
    Returns:
    - Y: Result of the iterative operation.
    """
    binary_image = np.where(arr > threshold, 1, 0)  # Convert to binary
    P, Q = B.shape  # Structuring element size

    # Initialize X0 with the starting point p
    Xk = np.zeros_like(binary_image)
    Xk[p[0], p[1]] = 1  # Set the point p in X0

    while True:
        # Dilate Xk-1
        padded_Xk = np.pad(Xk, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='constant', constant_values=0)
        dilated = np.zeros_like(Xk)

        for i in range(P // 2, padded_Xk.shape[0] - P // 2):
            for j in range(Q // 2, padded_Xk.shape[1] - Q // 2):
                region = padded_Xk[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
                if np.any(region[B == 1] == 1):  # If any pixel in the region matches the structuring element
                    dilated[i - P // 2, j - Q // 2] = 1

        # Compute the intersection with A
        Xk_new = np.logical_and(dilated, binary_image).astype(int)

        # Check for convergence
        if np.array_equal(Xk, Xk_new):
            break

        Xk = Xk_new  # Update for the next iteration

    return Xk


def convert_to_grayscale_and_binary(arr, threshold=127):
    """Convert a grayscale image to binary (0 or 1) using the given threshold."""
    return np.where(arr > threshold, 1, 0)


def get_neighborhood(arr, x, y, B):
    """Get the neighborhood pixels of (x, y) based on structuring element B."""
    rows, cols = arr.shape
    neighborhood = []
    B_rows, B_cols = B.shape

    for i in range(B_rows):
        for j in range(B_cols):
            nx = x + i - B_rows // 2
            ny = y + j - B_cols // 2
            if 0 <= nx < rows and 0 <= ny < cols:
                if B[i, j] == 1:
                    neighborhood.append((nx, ny))

    return neighborhood


def print_help():
    help_text = """
        Usage: python3 task3.py [command] [param]

        Commands:
            --help                     Show this help message

            --dilation <param>         Apply dilation with the specified param (e.g., structuring element size)
            --erosion <param>          Apply erosion with the specified param (e.g., structuring element size)
            --opening <param>          Apply opening with the specified param (e.g., structuring element size)
            --closing <param>          Apply closing with the specified param (e.g., structuring element size)
            --hmt <param>              Apply hit-or-miss transformation with the specified param (e.g., structuring element size)
            --iterative <param> <p>    Apply iterative morphological operation with the specified param and point p
            
        Examples:
            python3 task3.py --dilation 3
            python3 task3.py --erosion 3
            python3 task3.py --opening 3
            python3 task3.py --closing 3
            python3 task3.py --hmt 3
            python3 task3.py --iterative 3 10,10
    """
    print(help_text)


if len(sys.argv) < 3:
    print("Too few command line parameters given.\n")
    print_help()
    sys.exit()

command = sys.argv[1]
param = sys.argv[2]

# Load the image
#image_path = "./components/images/lenac.bmp"
image_path = "./components/images/g_lena_small.bmp"

image = Image.open(image_path)
image = image.convert("L")
arr = np.array(image)

if command == '--help':
    print_help()
    sys.exit()

if command == '--dilation':
    result_arr = dilation(np.ones((int(param), int(param)), dtype=int), arr)
elif command == '--erosion':
    result_arr = erosion(np.ones((int(param), int(param)), dtype=int), arr)
elif command == '--opening':
    result_arr = opening(np.ones((int(param), int(param)), dtype=int), arr)
elif command == '--closing':
    result_arr = closing(np.ones((int(param), int(param)), dtype=int), arr)
elif command == '--hmt':
    result_arr = hmt_transformation(np.ones((int(param), int(param)), dtype=int), arr)
elif command == '--iterative':
    p = tuple(map(int, sys.argv[3].split(',')))
    result_arr = iterative_morphological_operation(np.ones((int(param), int(param)), dtype=int), arr, p)
else:
    print(f"Unknown command: {command}")
    print_help()
    sys.exit()

new_im = Image.fromarray((result_arr * 255).astype(np.uint8))
new_im.save("result.bmp")
print(f"Result saved as 'result.bmp'.")
