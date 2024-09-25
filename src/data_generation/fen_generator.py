import chess
import random

def generate_random_fen():
    board = chess.Board()

    for _ in range(random.randint(10, 50)):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)

    return board.fen()

def generate_fen_dataset(num_fen):
    dataset = []
    for _ in range(num_fen):
        dataset.append(generate_random_fen())
    return dataset



#%%
