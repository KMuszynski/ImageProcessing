import numpy as np

from .utilities import pad_image, get_subregion


def erosion(A, B):
    """
    Perform erosion on binary image A with structuring element B.
    Erosion: (A ⊖ B)
    A and B are binary (0/1) arrays.
    """
    padded_A, p, q = pad_image(A, B)
    eroded = np.zeros_like(A)

    # For erosion: a pixel in A remains 1 if all B=1 positions are 1 in A
    # i.e. B acts like a mask that must fit entirely into A's foreground.
    for i in range(p, padded_A.shape[0] - p):
        for j in range(q, padded_A.shape[1] - q):
            region = get_subregion(padded_A, i, j, B)
            # Check if all pixels under B=1 are also 1 in the region
            if np.all(region[B == 1] == 1):
                eroded[i - p, j - q] = 1

    return eroded


def dilation(A, B):
    """
    Perform dilation on binary image A with structuring element B.
    Dilation: (A ⊕ B)
    """
    padded_A, p, q = pad_image(A, B)
    dilated = np.zeros_like(A)

    # For dilation: a pixel in the output is 1 if any of the pixels
    # under B=1 in the neighborhood is 1 in A.
    for i in range(p, padded_A.shape[0] - p):
        for j in range(q, padded_A.shape[1] - q):
            region = get_subregion(padded_A, i, j, B)
            if np.any(region[B == 1] == 1):
                dilated[i - p, j - q] = 1

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
    Perform erosion on binary image A with a 3-value structuring element B. Helper method for m4_operation (handling
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
