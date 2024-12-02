import cv2

IMG_SIZE = (64, 64)

def parse_fen(fen):
    """
    Parse un FEN pour générer un tableau 8x8 indiquant les cases occupées (1) ou vides (0).
    """
    rows = fen.split(" ")[0].split("/")  # Extraire uniquement la partie des positions
    board = []
    for row in rows:
        row_data = []
        for char in row:
            if char.isdigit():  # Les chiffres indiquent des cases vides consécutives
                row_data.extend([0] * int(char))
            else:  # Les caractères indiquent des pièces
                row_data.append(1)
        board.append(row_data)
    return np.array(board)


def split_board_image(image, fen):
    """
    Découper une image de plateau en 64 sous-images et générer leurs labels d'occupation.
    """
    board = parse_fen(fen)  # Générer le tableau 8x8 des labels
    cell_size_x = image.shape[1] // 8
    cell_size_y = image.shape[0] // 8

    images = []
    labels = []

    for i in range(8):
        for j in range(8):
            # Découper chaque case
            x_start, x_end = j * cell_size_x, (j + 1) * cell_size_x
            y_start, y_end = i * cell_size_y, (i + 1) * cell_size_y
            cell = image[y_start:y_end, x_start:x_end]
            # Redimensionner la case
            cell = cv2.resize(cell, IMG_SIZE)
            images.append(cell)
            labels.append(board[i, j])  # 1 = occupé, 0 = vide

    return images, labels
