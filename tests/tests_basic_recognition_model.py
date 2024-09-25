import pytest

from src.basic_recognition_model.board_operations import generate_fen

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
