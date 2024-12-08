import numpy as np


def to_binary(arr, threshold=127):
    """
    Convert a grayscale image array to binary (0/1) using a given threshold.
    """
    return (arr > threshold).astype(np.uint8)


def pad_image(img_arr, B):
    """
    Pad the image based on the size of the structuring element B.
    B should be a binary array (0/1).
    """
    # Compute needed padding
    p = B.shape[0] // 2
    q = B.shape[1] // 2
    return np.pad(img_arr, ((p, p), (q, q)), mode='constant', constant_values=0), p, q


def get_subregion(padded_img, i, j, B):
    """
    Extract the subregion of padded_img corresponding to the center at (i, j)
    that aligns with B. Used internally by morphological operations.
    """
    P, Q = B.shape
    return padded_img[i - P // 2: i + P // 2 + 1, j - Q // 2: j + Q // 2 + 1]
