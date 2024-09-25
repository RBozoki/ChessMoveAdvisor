import chess
import random

# Génère une position aléatoire valide en jouant des coups aléatoires
def generate_random_fen():
    board = chess.Board()

    # On joue un nombre aléatoire de coups, ici entre 10 et 50 coups
    for _ in range(random.randint(10, 50)):
        if board.is_game_over():
            break
        # Sélectionne un coup aléatoire parmi les coups légaux
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)

    return board.fen()

# Générer un nombre donné de FEN
def generate_fen_dataset(num_fen):
    dataset = []
    for _ in range(num_fen):
        dataset.append(generate_random_fen())
    return dataset



#%%
