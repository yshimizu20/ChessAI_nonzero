import torch
import torch.nn as nn
import chess
import random

from chess_engine.model.policy import PolicyNetwork
from chess_engine.model.value import ValueNetwork
from chess_engine.utils.state import createStateObj
from chess_engine.utils.dataloader import DataLoaderCluster, TestLoader

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
batch_size = 200
num_games = 200

cluster = DataLoaderCluster()
testing_iterator = TestLoader("filtered/db2023.pgn")


def self_play(policynet, valuenet):
    data = []

    for i in range(num_games):
        board = chess.Board()
        states, moves = [], []

        while not board.is_game_over():
            state = createStateObj(board)
            states.append(state)

            legal_moves = board.legal_moves
            legal_mask = torch.zeros(1968, dtype=torch.float32)
            for move in legal_moves:
                legal_mask[move.from_square * 64 + move.to_square] = 1.0

            policy = policynet(state, legal_mask)
            # select top 10 moves
            top10 = torch.topk(policy, 10)[1]

            # select best move based on value network
            best_move = None
            best_value = -10
            for move in top10:
                board.push(move)
                value = valuenet(createStateObj(board))
                if value > best_value:
                    best_move = move
                    best_value = value
                board.pop()

            moves.append(best_move)
            board.push(best_move)

        # get winner
        winner = 0.0
        if board.result() == "1-0":
            winner = 1.0
        elif board.result() == "0-1":
            winner = -1.0

        # add to data
        for i in range(len(states)):
            data.append((states[i], moves[i], winner))

    random.shuffle(data)
    X, y, win = zip(*data)
    return X, y, win
