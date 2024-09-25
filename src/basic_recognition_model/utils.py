from pathlib import Path
from src.basic_recognition_model.image_processing import load_image_as_vector

def load_pieces():
    pieces = []
    labels = []

    base_path = Path(__file__).resolve().parent.parent.parent / "pieces" / "piece_set_name"

    piece_colors = [("Black", base_path / "black"),
                    ("White", base_path / "white")]

    for color, folder in piece_colors:
        # Utilise pathlib pour lister les fichiers .png
        for piece_file in folder.glob("*.png"):
            image_vector = load_image_as_vector(piece_file)
            pieces.append(image_vector)
            labels.append(f"{color}_{piece_file.stem}")

    black_square = load_image_as_vector(base_path / "black_square.png")
    white_square = load_image_as_vector(base_path / "white_square.png")

    pieces.append(black_square)
    labels.append("Empty")

    pieces.append(white_square)
    labels.append("Empty")

    return pieces, labels