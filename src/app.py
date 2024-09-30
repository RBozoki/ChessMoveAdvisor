import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from src.basic_recognition_model.board_operations import generate_fen
from src.basic_recognition_model.main import basic_recognition
from src.best_move_evaluation.evaluate import best_move

app = Flask(__name__)

# Dossier pour stocker les images téléchargées temporairement
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fonction pour vérifier les extensions de fichier
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Vérifier si le fichier fait partie de la requête
    if 'image' not in request.files:
        return "Pas de fichier d'image fourni", 400

    file = request.files['image']

    # Si aucun fichier n'est sélectionné
    if file.filename == '':
        return "Aucun fichier sélectionné", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Sauvegarder l'image dans le dossier temporaire
        file.save(file_path)

        # Obtenir les autres paramètres du formulaire
        turn = request.form.get('turn', 'w')
        white_king_castling = 'white_king_castling' in request.form
        white_queen_castling = 'white_queen_castling' in request.form
        black_king_castling = 'black_king_castling' in request.form
        black_queen_castling = 'black_queen_castling' in request.form

        # Processer l'image téléchargée pour reconnaître le plateau
        board = basic_recognition(file_path)
        fen = generate_fen(board)

        # Évaluer le meilleur coup
        best_move_result = best_move(fen, turn, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling)

        return f"Best move: {best_move_result}"

    return "Fichier non supporté", 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
