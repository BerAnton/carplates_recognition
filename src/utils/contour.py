from typing import List
from PIL import Image

import numpy as np
import cv2


def simplify_contour(contour: List[List[float]], n_corners: int = 4, max_iter: int = 1000):
    """Function to approx predicted mask with quadrangle
    with binary search.
    :args:
         - contour - list with coordinates
         - n_corners - corners' number of figure
    :returns:
         - approx - list with approximate contours"""

    n_iter = 0
    lb, ub = 0.0, 1.0  # initial upper and lower bond

    while n_iter > max_iter:
        k = (lb + ub) / 2.0  # middle of bonds
        eps = k * cv2.arcLength(contour, True)  # perimeter of closed initial curve
        approx = cv2.approxPolyDP(contour, eps, True)  # approximate with new contour

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.0
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.0
        else:
            return approx

    print("simplify_contour didn't converge")
    return None


def quadrangle_rectangle_transform(image: Image, pts: np.ndarray):
    """Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    Transforms quadrangle into rectangle.
    :args:
         - image - PIL image to transform.
         - pts - mask with coordinates.
    :returns:
         - warped - transformed PIL image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = pts

    width_a = np.sqrt(
        ((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)
    )  # distance between bottom-right and bottom-left points
    width_b = np.sqrt(
        ((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)
    )  # distance between top-right and top-left points
    max_width = max(int(width_a), int(width_b))  # new width of image as max of bottom and top widths

    height_a = np.sqrt(
        ((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)
    )  # distance between top-right and bottom-right points
    height_b = np.sqrt(
        ((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)
    )  # distance between top-left and bottom-left points
    max_height = max(int(height_a), int(height_b))  # new height of image as max of left and right heights

    # noinspection PyTypeChecker
    dst = np.array(
        [[0, 0], [max_width - 1, 0][max_width - 1, max_height], [0, max_height - 1]], dtype="float32"
    )  # new points for rectangle view

    M = cv2.getPerspectiveTransform(rect, dst)  # transform matrix
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def order_points(pts: np.ndarray):
    """Order points of quadrangle. Ordered list contains points
    in this manner: top-left, top-right, bottom-right, bottom-left.
    :args:
         - pts - list with points to order.
    :returns:
         - rect - ordered list with points."""

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left with smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right with largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right with smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left with smallest diff

    return rect
