import chess
import chess.pgn
import numpy as np
import torch
import itertools

from chess_engine.utils.utils import uci_dict, uci_table, piece_values


def createStateObj(board):
    # convert state into a 35 x 8 x 8 tensor
    state = torch.zeros((35, 8, 8), dtype=torch.float32)

    piece_map = board.piece_map()
    piece_types = [
        piece.piece_type - 1 if piece.color == chess.WHITE else piece.piece_type + 5
        for piece in board.piece_map().values()
    ]
    squares = [square for square in piece_map.keys()]
    state[
        piece_types,
        [square // 8 for square in squares],
        [square % 8 for square in squares],
    ] = 1.0
    net_piece_value = sum([piece_values[str(piece)] for piece in piece_map.values()])

    # append the 5 states above
    p1_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
    p1_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
    p2_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)
    p2_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
    turn = board.turn

    state[12, :, :] = float(p1_can_castle_queenside)
    state[13, :, :] = float(p1_can_castle_kingside)
    state[14, :, :] = float(p2_can_castle_queenside)
    state[15, :, :] = float(p2_can_castle_kingside)
    state[16, :, :] = float(turn)
    state[17, :, :] = float(net_piece_value)

    # enpassant
    if board.ep_square is not None:
        state[18, board.ep_square // 8, board.ep_square % 8] = 1.0

    # squares white pawns can move to
    from_mask = np.zeros(64, dtype=np.bool_)
    from_mask[list(board.pieces(chess.PAWN, chess.WHITE))] = 1
    from_mask = int.from_bytes(np.packbits(from_mask), byteorder="big")
    to_squares = [
        move.to_square for move in board.generate_legal_moves(from_mask=from_mask)
    ]
    state[19, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black pawns can move to
    from_mask = np.zeros(64, dtype=np.bool_)
    from_mask[list(board.pieces(chess.PAWN, chess.BLACK))] = 1
    from_mask = int.from_bytes(np.packbits(from_mask), byteorder="big")
    to_squares = [
        move.to_square for move in board.generate_legal_moves(from_mask=from_mask)
    ]
    state[20, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white pawns can attack
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.PAWN, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[21, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black pawns can attack
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.PAWN, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[22, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white knights can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.KNIGHT, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[23, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black knights can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.KNIGHT, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[24, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white bishops can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.BISHOP, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[25, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black bishops can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.BISHOP, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[26, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white rooks can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.ROOK, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[27, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black rooks can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.ROOK, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[28, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white queens can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.QUEEN, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[29, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black queens can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.QUEEN, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[30, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares white king can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.KING, chess.WHITE))
    ]
    to_squares = list(itertools.chain(*lst))
    state[31, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # squares black king can move to
    lst = [
        list(board.attacks(square))
        for square in list(board.pieces(chess.KING, chess.BLACK))
    ]
    to_squares = list(itertools.chain(*lst))
    state[32, [sq // 8 for sq in to_squares], [sq % 8 for sq in to_squares]] += 1.0

    # number of white attackers on each square
    state[33, :, :] = torch.tensor(
        [float(len(list(board.attackers(chess.WHITE, square)))) for square in range(64)]
    ).reshape(8, 8)

    # number of black attackers on each square
    state[34, :, :] = torch.tensor(
        [float(len(list(board.attackers(chess.BLACK, square)))) for square in range(64)]
    ).reshape(8, 8)

    return state


tag = {"1/2-1/2": 0.0, "1-0": 1.0, "0-1": -1.0}


def createData(fp, n_data=200):
    X, y, win = [], [], []

    for i in range(n_data):
        game = chess.pgn.read_game(fp)
        if game is None:
            if len(X) == 0:
                return [], [], []
            return X, y, win

        try:
            w = tag[game.headers["Result"]]
        except KeyError:
            continue

        board = game.board()

        lst = list(game.mainline_moves())
        len_lst = len(lst)
        for move in lst[:-1]:
            X.append(createStateObj(board))
            board.push(move)
            y_ele = torch.zeros(1968, dtype=torch.float32)
            try:
                y_ele[uci_dict[str(move)]] = 1.0
            except KeyError:
                X.pop()
                break
            y.append(y_ele)
            win.append(torch.tensor(w * (i + 1) / len_lst, dtype=torch.float32))

    return X, y, win
