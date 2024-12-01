import cv2
import numpy as np


def detect_board_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Collecter les points d'intersection des lignes
    corners = []
    if lines is not None:
        for line1 in lines:
            for line2 in lines:
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]
                intersection = get_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                if intersection:
                    corners.append(intersection)

    # Trouver les 4 coins les plus probables
    corners = refine_corners(corners, image)
    return corners

def get_intersection(line1, line2):
    # Calcul de l'intersection entre deux lignes
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Pas d'intersection
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

def refine_corners(corners, image):
    # Filtrer et ordonner les coins (par exemple avec un clustering type DBSCAN)
    # Placeholder pour simplifier
    corners = sorted(corners, key=lambda p: (p[0], p[1]))[:4]
    return corners
