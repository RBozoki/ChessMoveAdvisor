from src.recognition_model.detect_board import detect_board_corners
from src.recognition_model.extract_squares import extract_squares
from src.recognition_model.warp_board import warp_board



def process_image(image):
    corners = detect_board_corners(image)
    if len(corners) != 4:
        raise ValueError("Could not detect all corners.")
    warped = warp_board(image, corners)
    squares = extract_squares(warped)

    # Passe chaque case dans les modèles
    occupancies = [occupancy_model(square) for square in squares]
    pieces = [piece_model(square) if occ == 1 else None for occ, square in zip(occupancies, squares)]

    return occupancies, pieces
