from flask import Flask, render_template, request
from src.basic_recognition_model.board_operations import generate_fen
from src.basic_recognition_model.main import basic_recognition
from src.best_move_evaluation.evaluate import best_move

app = Flask(__name__)

def validate_turn_input(turn):
    return 'w' if turn == 'on' else 'b'

def validate_castling_input(castling):
    return castling == 'y'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    image_path = request.form['image_path']

    # Handle turn input from switch
    turn = 'w' if 'turn' in request.form else 'b'

    # Handle castling rights from switches (checkboxes)
    white_king_castling = 'white_king_castling' in request.form
    white_queen_castling = 'white_queen_castling' in request.form
    black_king_castling = 'black_king_castling' in request.form
    black_queen_castling = 'black_queen_castling' in request.form

    # Process image and generate FEN
    board = basic_recognition(image_path)
    fen = generate_fen(board)

    # Evaluate the best move
    best_move_result = best_move(fen, turn, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling)

    return f"Best move: {best_move_result}"

if __name__ == '__main__':
    app.run(debug=True)
