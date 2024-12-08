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
    Hit-or-Miss Transform.

    A: binary image
    B1, B2: binary structuring elements for foreground and background respectively.
    B1 marks the pattern to find in the foreground,
    B2 marks the pattern to find in the background.

    A^c is the complement of A.
    """

    # Erode A by B1 (all B1=1 must fit into A=1)
    A_erosion = erosion(A, B1)

    # Erode A^c by B2 (all B2=1 must fit into A^c=1)
    A_comp = 1 - A
    A_comp_erosion = erosion(A_comp, B2)

    # Intersection
    return A_erosion & A_comp_erosion
