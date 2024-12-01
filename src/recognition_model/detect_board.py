import cv2
import numpy as np

import os

from matplotlib import pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"


def detect_board_with_debug(image_path):
    # Charger l'image et la convertir en niveaux de gris
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Étape 1 : Détecter les bords
    edges = cv2.Canny(gray, 70, 180, apertureSize=3)
    plt.imshow(edges, cmap='gray')
    plt.show()

    # Étape 2 : Détection de lignes avec la transformation de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        raise ValueError("Aucune ligne détectée. Vérifiez la qualité de l'image.")

    # Dessiner les lignes détectées
    line_image = image.copy()
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #cv2.imshow("Lignes détectées (Hough)", line_image)

    # Étape 3 : Regrouper les lignes en horizontales et verticales
    horizontal_lines, vertical_lines = [], []
    for rho, theta in lines[:, 0]:
        if np.sin(theta) > 0.9:  # Lignes verticales
            vertical_lines.append((rho, theta))
        elif np.cos(theta) > 0.9:  # Lignes horizontales
            horizontal_lines.append((rho, theta))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        raise ValueError("Nombre insuffisant de lignes pour détecter le plateau.")

    # Étape 4 : Trouver les intersections
    intersections = []
    for rho_h, theta_h in horizontal_lines[:2]:
        for rho_v, theta_v in vertical_lines[:2]:
            a_h, b_h = np.cos(theta_h), np.sin(theta_h)
            a_v, b_v = np.cos(theta_v), np.sin(theta_v)
            x = (b_h * rho_v - b_v * rho_h) / (a_h * b_v - a_v * b_h)
            y = (a_v * rho_h - a_h * rho_v) / (a_h * b_v - a_v * b_h)
            intersections.append((int(x), int(y)))

    if len(intersections) != 4:
        raise ValueError("Impossible de détecter les quatre coins du plateau.")

    # Dessiner les intersections
    intersections_image = image.copy()
    for point in intersections:
        cv2.circle(intersections_image, point, 10, (0, 0, 255), -1)
    #cv2.imshow("Intersections détectées", intersections_image)

    # Étape 5 : Transformation perspective
    ordered_points = np.float32([
        intersections[0],  # Haut-gauche
        intersections[1],  # Haut-droit
        intersections[2],  # Bas-gauche
        intersections[3]   # Bas-droit
    ])
    target_points = np.float32([
        [0, 0],
        [400, 0],
        [0, 400],
        [400, 400]
    ])
    matrix = cv2.getPerspectiveTransform(ordered_points, target_points)
    board_warped = cv2.warpPerspective(image, matrix, (400, 400))
    #cv2.imshow("Plateau redressé (perspective)", board_warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return board_warped
# Utilisation
image_path = "./../../cog_data/train/0008.png"  # Remplace par le chemin de ton image
board = detect_board_with_debug(image_path)

