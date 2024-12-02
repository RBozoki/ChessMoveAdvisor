import cv2
import numpy as np

import os

from matplotlib import pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"


def detect_board(image_path):
    # Charger l'image et la convertir en niveaux de gris
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Étape 1 : Détecter les bords
    edges = cv2.Canny(gray, 70, 180, apertureSize=3)
    plt.imshow(edges, cmap='gray')
    plt.show()

    # Étape 2 : Détection de lignes avec la transformation de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)
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
    #plt.imshow(line_image, cmap='gray')
    #plt.show()

    # Étape 3 : Regrouper les lignes en horizontales et verticales
    horizontal_lines, vertical_lines = [], []
    for rho, theta in lines[:, 0]:
        if np.sin(theta) > 0.8:  # Lignes verticales
            vertical_lines.append((rho, theta))
        elif np.cos(theta) > 0.8:  # Lignes horizontales
            horizontal_lines.append((rho, theta))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        raise ValueError("Nombre insuffisant de lignes pour détecter le plateau.")

    # Étape 4 : Trouver les intersections
    height, width = gray.shape  # Dimensions de l'image

    intersections = []
    for rho_h, theta_h in horizontal_lines:
        for rho_v, theta_v in vertical_lines:
            a_h, b_h = np.cos(theta_h), np.sin(theta_h)
            a_v, b_v = np.cos(theta_v), np.sin(theta_v)
            determinant = a_h * b_v - a_v * b_h
            if determinant == 0:  # Lignes parallèles
                continue
            x = (b_h * rho_v - b_v * rho_h) / determinant
            y = (a_v * rho_h - a_h * rho_v) / determinant
            # Appliquer la négation pour corriger les coordonnées négatives
            intersections.append((int(-x), int(-y)))

    # Vérifier qu'il y a des intersections
    if not intersections:
        print("Aucune intersection détectée.")
        return

    # Sélectionner les 4 coins extrêmes
    intersections = np.array(intersections)
    top_left = intersections[np.argmin(intersections[:, 0] + intersections[:, 1])]
    top_right = intersections[np.argmax(intersections[:, 0] - intersections[:, 1])]
    bottom_right = intersections[np.argmax(intersections[:, 0] + intersections[:, 1])]
    bottom_left = intersections[np.argmin(intersections[:, 0] - intersections[:, 1])]

    height = bottom_left[1] - top_left[1]
    width = bottom_right[0] - bottom_left[0]

    if height > width:
        bottom_right[0] = bottom_left[0] + width * 2.5
        top_right[0] = top_left[0] + width * 1.5

    bottom_right[1] -= 30


    corners = [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]

    print("Coins détectés :", corners)

    # Dessiner les coins sur l'image
    corners_image = image.copy()
    for point in corners:
        cv2.circle(corners_image, (point[0], point[1]), 10, (0, 0, 255), -1)

    #plt.imshow(corners_image, cmap='gray')
    #plt.show()


    # Étape 5 : Transformation perspective
    ordered_points = np.float32(corners)  # Coins détectés (ordre haut-gauche, haut-droit, bas-gauche, bas-droit)
    target_points = np.float32([
        [0, 0],
        [400, 0],
        [0, 400],
        [400, 400]
    ])  # Grille cible de 400x400 pixels

    # Calcul de la matrice de transformation
    matrix = cv2.getPerspectiveTransform(ordered_points, target_points)
    board_warped = cv2.warpPerspective(image, matrix, (400, 400))

    # Afficher le plateau redressé
    #plt.imshow(cv2.cvtColor(board_warped, cv2.COLOR_BGR2RGB))
    #plt.show()

    return board_warped

# Utilisation

image_path = "./../../cog_data/train/0008.png"  # Remplace par le chemin de ton image
board = detect_board(image_path)

