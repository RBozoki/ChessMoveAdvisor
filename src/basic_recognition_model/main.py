from src.basic_recognition_model.utils import load_pieces
from src.basic_recognition_model.knn_model import train_knn, predict_with_threshold
from src.basic_recognition_model.image_processing import divide_image_into_vectors
from src.basic_recognition_model.board_operations import print_chess_board, generate_fen

def basic_recognition(image_path):
    pieces, labels = load_pieces()
    knn = train_knn(pieces, labels)

    vectors = divide_image_into_vectors(image_path)

    predictions = []
    for vec in vectors:
        vec = vec.reshape(1, -1)
        prediction, distance = predict_with_threshold(knn, vec, max_empty_distance=500)
        predictions.append(prediction)

    return predictions


if __name__ == "__main__":
    image_path = "./../../data/chess_position_0.png"
    board = basic_recognition(image_path)
    print_chess_board(board)
    fen = generate_fen(board)
    print(fen)
