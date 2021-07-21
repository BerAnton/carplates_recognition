from typing import Tuple
from PIL import Image

import cv2


class Resize(object):
    def __init__(self, size: Tuple[int, int] = (320, 64)) -> None:
        self.size = size

    def __call__(self, img: Image):
        w_from, h_from = img.shape[1], img.shape[0]
        w_to, h_to = self.size

        interpolation = cv2.INTER_AREA
        if w_to > w_from:
            interpolation = cv2.INTER_CUBIC

        img = cv2.resize(img, dsize=self.size, interpolation=interpolation)
        return img
