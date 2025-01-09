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


def erosion_custom(A, B):
    """
    Perform erosion on binary image A with a 3-value structuring element B.
    Values in B:
    - 1: require A=1 at that position
    - 0: require A=0 at that position
    - 2: inactive/don't care (no requirement)
    """
    # Pad the input image to handle boundaries
    padded_A, pad_y, pad_x = pad_image(A, B)
    eroded = np.zeros_like(A)

    # Iterate over every pixel in the original (unpadded) image
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            # Extract region from the padded image
            region = get_subregion(padded_A, y + pad_y, x + pad_x, B)

            # Check matching conditions
            fg_required = (B == 1)  # Positions in B where A=1 is required
            bg_required = (B == 0)  # Positions in B where A=0 is required

            # All fg_required positions must match A=1, and all bg_required positions must match A=0
            if np.all(region[fg_required] == 1) and np.all(region[bg_required] == 0):
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


def hit_or_miss_custom(A, B1, B2):
    """
    Hit-or-Miss Transform with inactive pixels.

    A: Binary input image.
    B1: Structuring element for the foreground (values in {0, 1, 2}).
    B2: Structuring element for the background (values in {0, 1, 2}).
    """
    # Compute the complement of A
    A_comp = 1 - A

    # Perform custom erosion with B1 and B2
    A_eroded = erosion_custom(A, B1)
    A_comp_eroded = erosion_custom(A_comp, B2)

    # Intersection
    return A_eroded & A_comp_eroded



def m4_operation_hmt(A, B1_list, max_iterations=1000):
    """
    For each 3-value structuring element B1_i in B1_list:
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
            hmt_res = hit_or_miss_custom(X_old, B1_i, B2_i)

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
