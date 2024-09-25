import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from src.basic_recognition_model.board_operations import generate_fen

# tests_board_operations

def test_generate_fen():
    # Test d'un plateau d'échecs complet
    board_list = [
        'Black_Rook', 'Black_Knight', 'Black_Bishop', 'Black_Queen', 'Black_King', 'Black_Bishop', 'Black_Knight', 'Black_Rook',
        'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn',
        'White_Rook', 'White_Knight', 'White_Bishop', 'White_Queen', 'White_King', 'White_Bishop', 'White_Knight', 'White_Rook'
    ]

    fen = generate_fen(board_list)
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fen == expected_fen


def test_generate_fen_with_additional_info():
    # Test avec des informations supplémentaires
    board_list = [
        'Black_Rook', 'Black_Knight', 'Black_Bishop', 'Black_Queen', 'Black_King', 'Black_Bishop', 'Black_Knight', 'Black_Rook',
        'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn', 'Black_Pawn',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty',
        'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn', 'White_Pawn',
        'White_Rook', 'White_Knight', 'White_Bishop', 'White_Queen', 'White_King', 'White_Bishop', 'White_Knight', 'White_Rook'
    ]

    fen = generate_fen(board_list, add_info=True, turn='b', castling_rights='KQk', en_passant='e3', halfmove_clock=4, fullmove_number=10)
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQk e3 4 10"
    assert fen == expected_fen


def test_generate_fen_invalid_length():
    # Test de vérification de la longueur incorrecte de la liste
    board_list = ['Empty'] * 63  # Liste trop courte
    with pytest.raises(AssertionError, match="La liste doit contenir exactement 64 éléments."):
        generate_fen(board_list)


# tests_image_processing

from src.basic_recognition_model.image_processing import load_image_as_vector, divide_image_into_vectors

# Helper function to create an image in memory
def create_test_image(size=(800, 800), mode='RGB', color=(255, 255, 255)):
    """Create an image with the given size, mode, and color."""
    image = Image.new(mode, size, color)
    return image

def test_load_image_as_vector_rgb():
    # Créer une image de test RGB
    image = create_test_image(size=(100, 100), mode='RGB', color=(255, 0, 0))

    # Sauvegarder l'image dans un buffer mémoire
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Charger l'image via la fonction
    vector = load_image_as_vector(buffer)

    # Vérifier la taille du vecteur (100*100*3 = 30000 pixels car mode RGB)
    assert vector.shape == (100 * 100 * 3,)
    assert np.array(vector).sum() == 255 * 100 * 100  # Le rouge est 255 pour chaque pixel

def test_load_image_as_vector_rgba():
    # Créer une image de test RGBA
    image = create_test_image(size=(100, 100), mode='RGBA', color=(255, 0, 0, 128))

    # Sauvegarder l'image dans un buffer mémoire
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Charger l'image via la fonction
    vector = load_image_as_vector(buffer)

    # Vérifier la taille du vecteur après conversion en RGB (100*100*3 = 30000 pixels)
    assert vector.shape == (100 * 100 * 3,)
    assert np.array(vector).sum() == 255 * 100 * 100  # Le rouge est 255 pour chaque pixel

def test_load_image_as_vector_invalid_size():
    # Créer une image de test de taille différente
    image = create_test_image(size=(50, 50), mode='RGB', color=(255, 0, 0))

    # Sauvegarder l'image dans un buffer mémoire
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Charger l'image via la fonction
    vector = load_image_as_vector(buffer)

    # Vérifier que la taille du vecteur est correcte après redimensionnement (100*100*3 = 30000)
    assert vector.shape == (100 * 100 * 3,)

def test_divide_image_into_vectors():
    # Créer une image 800x800
    image = create_test_image(size=(800, 800), mode='RGB', color=(255, 0, 0))

    # Sauvegarder l'image dans un buffer mémoire
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Diviser l'image en vecteurs
    vectors = divide_image_into_vectors(buffer)

    # Vérifier qu'il y a 64 vecteurs (8x8 cases)
    assert len(vectors) == 64

    # Vérifier que chaque vecteur a la taille correcte (100*100*3 = 30000)
    for vector in vectors:
        assert vector.shape == (100 * 100 * 3,)
        assert np.array(vector).sum() == 255 * 100 * 100  # Le rouge est 255 pour chaque pixel

def test_divide_image_into_vectors_invalid_size():
    # Créer une image de taille incorrecte
    image = create_test_image(size=(500, 500), mode='RGB', color=(255, 0, 0))

    # Sauvegarder l'image dans un buffer mémoire
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Tester si une assertion est levée pour une image de taille incorrecte
    with pytest.raises(AssertionError, match="L'image doit être de 800x800 pixels"):
        divide_image_into_vectors(buffer)
