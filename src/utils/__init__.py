""" The utils module contains utility functions. """
import cv2
import numpy as np


def create_view(frame):
    """Show current frame of the RGB-D dataset as images.

    :param frame:
    :return:
    """
    image, depth, norms, mask = frame

    # convert normals from [-1, 1] to [0, 255]
    norms = ((norms + 1) / 2) * 255

    # apply a colormap on grayscale depth map, makes easier to see depth changes
    depth = cv2.applyColorMap((depth * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    masked_image = image.copy()

    bg_color = 128  # gray window background
    masked_image[mask, :] = bg_color
    depth[mask, :] = bg_color
    norms[mask, :] = bg_color

    dst = np.hstack((image, masked_image, depth, norms))
    return dst.astype(np.uint8)
