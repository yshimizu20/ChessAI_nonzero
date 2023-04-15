import torch
import torch.nn as nn
import chess
import random
import math
from typing import List

from chess_engine.model.policy import PolicyNetwork
from chess_engine.model.value import ValueNetwork
from chess_engine.utils.state import createStateObj
from chess_engine.utils.dataloader import DataLoaderCluster, TestLoader
from chess_engine.utils.utils import uci_dict, uci_table

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
batch_size = 200
num_games = 200

cluster = DataLoaderCluster()
testing_iterator = TestLoader("filtered/db2023.pgn")


def MCTS(orig_board: chess.Board, policynet: PolicyNetwork) -> str:
    def uct(node: List):
        visits, value, *_ = node
        return value + math.sqrt(2 * math.log(visits) / visits)

    nodes = {}  # key: state_str, value: Tuple(value, n_visits, children)

    for _ in range(1000000):
        # select
        board_str = orig_board.fen()
        hist = []
        while nodes[board_str].children:
            hist.append(board_str)
            children = [nodes[child] for child in nodes[board_str].children]
            board_str = max(children, key=uct)

        board = chess.Board(board_str)

        if not board.is_game_over():
            # expand
            legal_moves = board.legal_moves
            children = []
            for move in legal_moves:
                board.push(move)
                child_str = board.fen()
                children.append(child_str)
                nodes[child_str] = [0, 0, []]
                board.pop()

            nodes[board_str] = [value, 1, children]
            hist.append(board_str)

            # rollout using policy network
            while not board.is_game_over():
                legal_moves = board.legal_moves
                legal_mask = torch.zeros(1968, dtype=torch.float32)
                for move in legal_moves:
                    legal_mask[uci_dict[move.uci()]] = 1.0

                policy = policynet(createStateObj(board), legal_mask)
                move = uci_table[torch.argmax(policy)]
                board.push(move)

        # backpropagate
        if board.result() == "1-0":
            value = 1
        elif board.result() == "0-1":
            value = -1
        elif board.result() == "1/2-1/2":
            value = 0
        else:
            raise ValueError("Invalid result")

        for state_str in hist[::-1]:
            node = nodes[state_str]
            node[0] += 1
            node[1] += value

    # select best move
    children = [nodes[child] for child in nodes[orig_board.fen()].children]
    best_move = max(children, key=lambda x: x[1])

        print(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {num_epochs}\n")
            fp.write(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        del X, y, win, policy, value

        if epoch % 10 == 0:
            # evaluate
            X, y, win = (
                testing_iterator.X.to(device),
                testing_iterator.y.to(device),
                testing_iterator.win.to(device),
            )

            with torch.no_grad():
                policy = policynet(X)
                value = valuenet(X)
                policy_loss = policy_criterion(policy, y)
                value_loss = value_criterion(value, win)

            print(f"Test Policy Loss: {policy_loss}, Test Value Loss: {value_loss}")
            with open(log_path, "a") as fp:
                fp.write(
                    f"Test Policy Loss: {policy_loss}, Test Value Loss: {value_loss}\n"
                )

            del X, y, win, policy, value

        # save model
        if epoch % 100 == 99:
            torch.save(policynet.state_dict(), f"saved_models/policy_{epoch}.pt")
            torch.save(valuenet.state_dict(), f"saved_models/value_{epoch}.pt")


def self_play(policynet: PolicyNetwork):
    data = []

    for i in range(num_games):
        board = chess.Board()
        states, moves = [], []

        while not board.is_game_over():
            state = createStateObj(board)
            states.append(state)

            next_best_state = MCTS(board, policynet)
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                if board.fen() == next_best_state:
                    best_move = move
                    break
                board.pop()

            if not best_move:
                raise ValueError("No best move found")
            moves.append(best_move)

        # get winner
        if board.result() == "1-0":
            winner = 1.0
        elif board.result() == "0-1":
            winner = -1.0
        elif board.result() == "1/2-1/2":
            winner = 0.0
        else:
            raise ValueError("Invalid result")

        # add to data
        for i in range(len(states)):
            data.append((states[i], moves[i], winner))

    random.shuffle(data)
    X, y, win = zip(*data)
    return X, y, win


def train_rl(
    start_epoch=0, end_epoch=5000, policy_model_path=None, value_model_path=None
):
    policynet = PolicyNetwork()
    valuenet = ValueNetwork()
    if policy_model_path is not None:
        policynet.load_state_dict(torch.load(policy_model_path))
    if value_model_path is not None:
        valuenet.load_state_dict(torch.load(value_model_path))
    policynet.to(device)
    valuenet.to(device)

    value_criterion = nn.MSELoss()
    # train value network only
    optimizer = torch.optim.Adam(valuenet.parameters(), lr=lr)
    log_path = "logs/rl_train.txt"

    for epoch in range(start_epoch, end_epoch):
        X, y, win = self_play(policynet, valuenet)

        X = torch.stack(X, dim=0).to(device)
        y = torch.stack(y, dim=0).to(device)
        win = torch.stack(win).unsqueeze(1).to(device)

        # train value network
        value_loss = 0
        value = valuenet(X)
        value_loss = value_criterion(value, win)

        # train both networks
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        # log
        print(f"Epoch {epoch + 1} of {end_epoch}\n")
        print(f"Value Loss: {value_loss.item()}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {end_epoch}\n")
            fp.write(f"Value Loss: {value_loss.item()}")

        # TODO: evaluate

        # save model
        if (epoch + 1) % 100 == 0:
            torch.save(policynet.state_dict(), f"models/rl_policy_{epoch}.pt")
