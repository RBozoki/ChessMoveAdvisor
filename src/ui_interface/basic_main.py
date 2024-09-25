from src.basic_recognition_model.board_operations import generate_fen
from src.basic_recognition_model.main import basic_recognition
from src.best_move_evaluation.evaluate import best_move


def validate_turn_input(turn):
    if turn.lower() not in ['w', 'b']:
        raise ValueError("Invalid input for turn. Please enter 'w' for white or 'b' for black.")
    return turn.lower()

def validate_castling_input(castling):
    if castling.lower() not in ['y', 'n']:
        raise ValueError("Invalid input for castling. Please enter 'y' for yes or 'n' for no.")
    elif castling.lower() == 'y':
        return True
    else:
        return False


image_path = input("Enter image path: ")
turn = validate_turn_input(input("Which turn? (w/b): "))
white_king_castling = validate_castling_input(input("Can white king castle? (y/n): "))
white_queen_castling = validate_castling_input(input("Can white queen castle? (y/n): "))
black_king_castling = validate_castling_input(input("Can black king castle? (y/n): "))
black_queen_castling = validate_castling_input(input("Can black queen castle? (y/n): "))


print(f"Image Path: {image_path}")
print(f"Turn: {turn}")
print(f"White King Castling: {white_king_castling}")
print(f"White Queen Castling: {white_queen_castling}")
print(f"Black King Castling: {black_king_castling}")
print(f"Black Queen Castling: {black_queen_castling}")

print("Processing image...")

board = basic_recognition(image_path)
fen = generate_fen(board)

print("Fen identified")
print("Evaluating best move...")

best_move = best_move(fen, turn, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling)

print(f"Best move: {best_move}")