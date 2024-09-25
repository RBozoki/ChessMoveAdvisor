def print_chess_board(board_list):
    assert len(board_list) == 64, "La liste doit contenir exactement 64 éléments."

    piece_symbols = {
        'Black_King': '♚', 'Black_Queen': '♛', 'Black_Rook': '♜', 'Black_Bishop': '♝', 'Black_Knight': '♞', 'Black_Pawn': '♟',
        'White_King': '♔', 'White_Queen': '♕', 'White_Rook': '♖', 'White_Bishop': '♗', 'White_Knight': '♘', 'White_Pawn': '♙',
        'Empty': '.'
    }

    for row in range(8):
        row_pieces = board_list[row * 8: (row + 1) * 8]
        row_symbols = [piece_symbols.get(piece, '?') for piece in row_pieces]
        print(' '.join(row_symbols))


def generate_fen(board_list, add_info=False, turn='w', castling_rights='KQkq', en_passant='-', halfmove_clock=0, fullmove_number=1):
    assert len(board_list) == 64, "La liste doit contenir exactement 64 éléments."

    piece_fen = {
        'Black_King': 'k', 'Black_Queen': 'q', 'Black_Rook': 'r', 'Black_Bishop': 'b', 'Black_Knight': 'n', 'Black_Pawn': 'p',
        'White_King': 'K', 'White_Queen': 'Q', 'White_Rook': 'R', 'White_Bishop': 'B', 'White_Knight': 'N', 'White_Pawn': 'P',
        'Empty': ''
    }

    fen_rows = []

    for row in range(8):
        fen_row = []
        empty_count = 0
        for col in range(8):
            piece = board_list[row * 8 + col]
            fen_symbol = piece_fen.get(piece, '')

            if fen_symbol == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row.append(str(empty_count))
                    empty_count = 0
                fen_row.append(fen_symbol)

        if empty_count > 0:
            fen_row.append(str(empty_count))

        fen_rows.append(''.join(fen_row))

    fen = '/'.join(fen_rows)

    if add_info:
        fen += f" {turn} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"

    return fen
