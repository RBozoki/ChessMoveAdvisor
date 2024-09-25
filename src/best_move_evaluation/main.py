from evaluate import best_move


fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"

if __name__ == "__main__":
    best_move = best_move(fen, 'b', True, True, True, True)
    print("Best move: ", best_move)