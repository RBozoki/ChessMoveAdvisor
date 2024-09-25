from flask import Flask, render_template, request
from src.basic_recognition_model.board_operations import generate_fen
from src.basic_recognition_model.main import basic_recognition
from src.best_move_evaluation.evaluate import best_move

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    image_path = request.form['image_path']
    turn = validate_turn_input(request.form['turn'])
    white_king_castling = validate_castling_input(request.form['white_king_castling'])
    white_queen_castling = validate_castling_input(request.form['white_queen_castling'])
    black_king_castling = validate_castling_input(request.form['black_king_castling'])
    black_queen_castling = validate_castling_input(request.form['black_queen_castling'])

    # Process image and generate FEN
    board = basic_recognition(image_path)
    fen = generate_fen(board)

    # Evaluate best move
    best_move_result = best_move(fen, turn, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling)

    return f"Best move: {best_move_result}"

if __name__ == '__main__':
    app.run(debug=True)
