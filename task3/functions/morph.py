import numpy as np

from .utilities import pad_image, get_subregion


def erosion(A, B):
    """
    Perform erosion on binary image A with structuring element B.
    """
    # Pad the input image to handle boundaries
    padded_A, pad_y, pad_x = pad_image(A, B)
    eroded = np.zeros_like(A)

    # Iterate over every pixel in the original (unpadded) image
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            # Extract region from the padded image
            region = get_subregion(padded_A, y + pad_y, x + pad_x, B)

            # Check if the structuring element B fits entirely
            if np.all(region[B == 1] == 1):  # Only consider "1" values in B
                eroded[y, x] = 1

    return eroded


def dilation(A, B):
    """
    Perform dilation on binary image A with structuring element B.
    Dilation: (A ⊕ B)
    """
    dilated = np.zeros_like(A)

    # Find coordinates of 1s in A
    coords_A = np.argwhere(A == 1)

    # Anchor point (center of B)
    anchor_y = B.shape[0] // 2
    anchor_x = B.shape[1] // 2

    # Iterate through each 1 in A
    for y, x in coords_A:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] == 1:
                    new_y = y + (i - anchor_y)  # Mapping row of the SE to the row of the A
                    new_x = x + (j - anchor_x)  # Mapping column of the SE to the column of A
                    if 0 <= new_y < A.shape[0] and 0 <= new_x < A.shape[1]:
                        dilated[new_y, new_x] = 1
    return dilated


def opening(A, B):
    """
    Opening: A ∘ B = (A ⊖ B) ⊕ B
    Used to remove small objects or noise from binary images.
    """
    return dilation(erosion(A, B), B)


def closing(A, B):
    """
    Closing: A ∙ B = (A ⊕ B) ⊖ B
    Used to fill small holes or gaps in binary images.
    """
    return erosion(dilation(A, B), B)


def hit_or_miss(A, B1, B2):
    """
    Standard Hit-or-Miss Transform.

    A: Binary input image.
    B1: Structuring element for the foreground.
    B2: Structuring element for the background.
    """
    # Compute the complement of A
    A_comp = 1 - A

    # Erode A using B1
    A_eroded = erosion(A, B1)

    # Erode A^c (complement of A) using B2
    A_comp_eroded = erosion(A_comp, B2)

    # Intersection
    return A_eroded & A_comp_eroded


def m4_operation_hmt(A, B1_list, max_iterations=1000):
    """
    For each 2 or 3-value structuring element B1_i in B1_list:
      - Create B2_i from B1_i
      - Let X_{i,0} = A
      - Repeat:
           X_{i,k} = hit_or_miss(X_{i,k-1}, B1_i, B2_i)  UNION A
        until X_{i,k} == X_{i,k-1} or max_iterations
      - Let D_i = stable X_{i,k}
    Then H(A) = D_1 ∪ D_2 ∪ ... ∪ D_n
    """
    H = np.zeros_like(A, dtype=np.uint8)

    for idx, B1_i in enumerate(B1_list, start=1):
        # Build B2_i
        B2_i = create_B2_from_B1(B1_i)

        X_old = A.copy()
        iteration_count = 0

        while True:
            # Perform hit-or-miss on X_old
            hmt_res = hit_or_miss(X_old, B1_i, B2_i)

            # Union with A
            X_new = np.logical_or(hmt_res, A).astype(np.uint8)

            print(f"[M4-HMT] SE {idx}, Iteration {iteration_count}, Foreground:", np.sum(X_new))

            # Check convergence
            if np.array_equal(X_new, X_old):
                break

            X_old = X_new
            iteration_count += 1
            if iteration_count >= max_iterations:
                print("Warning: Reached max_iterations without convergence.")
                break

        # Union stable result with H
        H = np.logical_or(H, X_old).astype(np.uint8)

    return H


def create_B2_from_B1(B1):
    """
    Convert B1 to B2:
    - B1=1 => B2=0
    - B1=0 => B2=1
    """
    B2 = B1.copy()
    B2[B1 == 1] = 0
    B2[B1 == 0] = 1

    return B2


def region_growing(arr, seed, threshold=10, region_value=1, connectivity=8):
    """
    Perform region growing segmentation on the image starting from a seed point.

    Parameters:
    - arr: Input grayscale or binary image (as a numpy array).
    - seed: Starting point for region growing (x, y).
    - threshold: Intensity difference threshold for adding neighboring pixels to the region (default is 10).
    - region_value: Value assigned to the segmented region (default is 1).
    - connectivity: Type of neighborhood connectivity (4-connected or 8-connected).

    Returns:
    - Segmented binary image with the region grown around the seed.
    """
    rows, cols = arr.shape
    segmented = np.zeros_like(arr)

    # Initialize a list for pixels to process (queue for BFS-like region growing)
    to_process = [seed]
    segmented[seed] = region_value

    while to_process:
        x, y = to_process.pop()

        # Check neighbors based on connectivity (4 or 8-connected)
        neighbors = get_neighbors(x, y, rows, cols, connectivity)

        for nx, ny in neighbors:
            if segmented[nx, ny] == 0:  # If not already in the region
                # Check if the intensity difference is below the threshold
                if abs(int(arr[x, y]) - int(arr[nx, ny])) <= threshold:
                    segmented[nx, ny] = region_value
                    to_process.append((nx, ny))

    return segmented.astype(np.uint8)


def get_neighbors(x, y, rows, cols, connectivity=8):
    """
    Get the neighboring pixels of (x, y) based on specified connectivity.

    Parameters:
    - x: The x-coordinate of the pixel.
    - y: The y-coordinate of the pixel.
    - rows: Number of rows in the image.
    - cols: Number of columns in the image.
    - connectivity: 4 or 8 connectivity (4-connected or 8-connected).

    Returns:
    - List of valid neighboring coordinates.
    """
    neighbors = []
    if connectivity == 8:  # 8-connected neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:  # 4-connected neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append((nx, ny))

    return neighbors


def merge_regions(segmented_img, threshold=10):
    """
    Merge adjacent regions that meet the similarity criterion.

    Parameters:
    - segmented_img: The segmented binary image.
    - threshold: Intensity difference threshold for merging regions.

    Returns:
    - Segmented image after region merging.
    """
    rows, cols = segmented_img.shape
    merged_img = segmented_img.copy()
    region_labels = np.unique(segmented_img)

    # Compare each pair of regions and merge if the difference is below threshold
    for i in range(rows):
        for j in range(cols):
            current_region = segmented_img[i, j]
            for nx, ny in get_neighbors(i, j, rows, cols, connectivity=8):
                neighbor_region = segmented_img[nx, ny]
                if current_region != 0 and neighbor_region != 0 and current_region != neighbor_region:
                    # Merge regions if their intensity difference is below the threshold
                    if abs(int(merged_img[i, j]) - int(merged_img[nx, ny])) <= threshold:
                        merged_img[nx, ny] = current_region

    return merged_img.astype(np.uint8)
