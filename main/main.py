import cv2
import numpy as np
def bw(image):
    if len(image.shape) == 3:
        gray = cv2.c