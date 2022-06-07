import cv2
import numpy as np


def depth2normals(depth):
    """Computes surface normals from a depth map.

    Args:
        depth (np.ndarray): A grayscale depth map image of size (H,W).

    Returns:
        np.ndarray: The corresponding surface normals map of size (H,W,3).
    """
    zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

    normals = np.dstack((-zx, -zy, np.ones_like(depth)))
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    normals[:, :, :] /= length

    # offset and rescale values to be in 0-1
    normals = (normals + 1) / 2
    return normals
