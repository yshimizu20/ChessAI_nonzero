import chess
import chess.pgn
import numpy as np
import random
import torch

from chess_engine.utils.utils import uci_dict, uci_table, piece_values


def createStateObj(board):
    # convert state into a 8x8x12 tensor
    state = torch.zeros((18, 8, 8), dtype=torch.float32)
    net_piece_value = 0

    for square, piece in board.piece_map().items():
        for square, piece in board.piece_map().items():
            net_piece_value += piece_values[str(piece)]
            if piece.color == chess.WHITE:
                state[piece.piece_type - 1, square // 8, square % 8] = 1.0
            else:
                state[piece.piece_type + 5, square // 8, square % 8] = 1.0

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

    return state


tag = ("1/2-1/2", "1-0", "0-1")


def createData(fp, n_data=200):
    data = []

    for i in range(n_data):
        X, y = [], []

        game = chess.pgn.read_game(fp)
        if game is None:
            if len(data) == 0:
                return [], [], []
            random.shuffle(data)
            X, y, win = zip(*data)
            return X, y, win

        if game.headers["Result"] == "1/2-1/2":
            win = 0.0
        elif game.headers["Result"] == "1-0":
            win = 1.0
        elif game.headers["Result"] == "0-1":
            win = -1.0
        else:
            print("Error: Unexpected result string" + game.headers["Result"])
            continue
        if game.headers["Result"] not in tag:
            continue

        board = game.board()

        lst = list(game.mainline_moves())
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

        for i in range(len(X)):
            data.append((X[i], y[i], torch.tensor(win, dtype=torch.float32)))

    random.shuffle(data)
    X, y, win = zip(*data)
    return X, y, win
