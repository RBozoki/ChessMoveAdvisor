import cv2
import numpy as np


def warp_board(image, corners):
    # Ordre attendu : [top-left, top-right, bottom-right, bottom-left]
    h, w = 800, 800  # Dimensions de l'image redressée
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    return warped