import numpy as np

from .utilities import pad_image, get_subregion


def erosion(A, B):
    """
    Perform erosion on binary image A with structuring element B.
    """
    eroded = np.zeros_like(A)

    # Find coordinates of 1s in A
    coords_A = np.argwhere(A == 1)

    # Anchor point (center of B)
    anchor_y, anchor_x = B.shape[0] // 2, B.shape[1] // 2

    # Iterate through each 1 in A
    for y, x in coords_A:
        is_match = True

        # Check all pixels in the structuring element B
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] == 1:
                    new_y = y + (i - anchor_y)  # Mapping row of the SE to the row of the A
                    new_x = x + (j - anchor_x)  # Mapping column of the SE to the column of A

                    # Fail if out of bounds or if no match
                    if not (0 <= new_y < A.shape[0] and 0 <= new_x < A.shape[1]) or A[new_y, new_x] != 1:
                        is_match = False
                        break
            if not is_match:
                break

        # Set the output pixel to 1 if checks passed
        if is_match:
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
    Hit-or-Miss Transform with inactives.

    A: binary image
    B1, B2: structuring elements with values in {0,1,2}
    - 1: foreground required
    - 0: background required
    - 2: inactive (don't care)

    A^c is the complement of A.
    """

    A_comp = 1 - A

    # Erode A by B1
    # B1=1 means we need A=1; B1=0 means we need A=0; B1=2 means no requirement.
    A_erosion = erosion_custom(A, B1)

    # Erode A^c by B2
    # Similarly, B2=1 means A^c=1 => A=0 required, B2=0 means A^c=0 => A=1 required, B2=2 no requirement.
    A_comp_erosion = erosion_custom(A_comp, B2)

    # Intersection
    return A_erosion & A_comp_erosion


def erosion_custom(A, B):
    """
    Perform erosion on binary image A with a 3-value structuring element B. Helper method for and hit_or_miss (handling
    inactive pixels).
    Values in B:
    1 -> require A=1 at that position
    0 -> require A=0 at that position
    2 -> inactive/don't care (no requirement)
    """
    padded_A, p, q = pad_image(A, B)
    eroded = np.zeros_like(A)

    for i in range(p, padded_A.shape[0] - p):
        for j in range(q, padded_A.shape[1] - q):
            region = get_subregion(padded_A, i, j, B)

            # Check only positions that matter
            fg_required = (B == 1)
            bg_required = (B == 0)

            # All fg_required must match A=1
            # All bg_required must match A=0
            if np.all(region[fg_required] == 1) and np.all(region[bg_required] == 0):
                eroded[i - p, j - q] = 1

    return eroded
