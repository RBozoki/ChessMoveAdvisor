import chess
import chess.engine
from pathlib import Path


def best_move(fen, turn, white_king_castling, white_queen_castling, black_king_castling, black_queen_castling):
    castling_rights = ''
    if white_king_castling:
        castling_rights += 'K'
    if white_queen_castling:
        castling_rights += 'Q'
    if black_king_castling:
        castling_rights += 'k'
    if black_queen_castling:
        castling_rights += 'q'
    if castling_rights == '':
        castling_rights = '-'

    complete_fen = f"{fen} {turn} {castling_rights} - 0 1"

    print(f"Request made with fen: {complete_fen}")

    stockfish_path = Path(__file__).resolve().parent.parent.parent / "stockfish" / "stockfish-ubuntu-x86-64-avx2"
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))

    board = chess.Board(complete_fen)

    result = engine.play(board, chess.engine.Limit(time=2.0))
    best_move = result.move

    engine.quit()

    return best_move
