import torch
import torch.nn as nn
import chess
import random

from chess_engine.model.policy import PolicyNetwork
from chess_engine.model.value import ValueNetwork
from chess_engine.utils.state import createStateObj
from chess_engine.utils.dataloader import DataLoader

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
batch_size = 200
start_epoch = 0
num_epochs = 5000
num_games = 200
log_path = f"log_{start_epoch}.txt"

training_iterators = [
    DataLoader(f"data/filtered/output-{year}_{month:02d}.pgn")
    for year in range(2015, 2018)
    for month in range(1, 13)
]
training_idx = 0
testing_iterator = DataLoader("data/filtered/db2023.pgn")


def train(data_source="dataset", policy_model_path=None, value_model_path=None):
    policynet = PolicyNetwork()
    valuenet = ValueNetwork()
    if policy_model_path is not None:
        policynet.load_state_dict(torch.load(policy_model_path))
    if value_model_path is not None:
        valuenet.load_state_dict(torch.load(value_model_path))
    policynet.to(device)
    valuenet.to(device)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    policy_optimizer = torch.optim.Adam(policynet.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(valuenet.parameters(), lr=lr)

    for epoch in range(start_epoch, num_epochs):
        if data_source == "self-play":
            X, y, win = self_play(policynet, valuenet)
        elif data_source == "dataset":
            try:
                X, y, win = training_iterators[training_idx].get_data(200)
            except StopIteration:
                training_idx += 1
                if training_idx == len(training_iterators):
                    training_idx = 0
                continue

        # train policy network
        policy_optimizer.zero_grad()
        policy_loss = 0
        # for state, move, value in zip(X, y, win):
        #     policy = policynet(state)
        #     policy_loss += policy_criterion(policy, move)
        policy = policynet(X)
        policy_loss += policy_criterion(policy, y)
        policy_loss.backward()
        policy_optimizer.step()

        # train value network
        value_optimizer.zero_grad()
        value_loss = 0
        # for state, move, value in zip(X, y, win):
        #     v = valuenet(state)
        #     value_loss += value_criterion(v, value)
        value = valuenet(X)
        value_loss += value_criterion(value, win)
        value_loss.backward()
        value_optimizer.step()

        print(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {num_epochs}\n")
            fp.write(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        if epoch % 10 == 0:
            # evaluate
            X, y, win = testing_iterator.get_data()
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

            # save model
            if epoch % 100 == 99:
                torch.save(policynet.state_dict(), f"saved_models/policy_{epoch}.pt")
                torch.save(valuenet.state_dict(), f"saved_models/value_{epoch}.pt")


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
        winner = 0
        if board.result() == "1-0":
            winner = 1
        elif board.result() == "0-1":
            winner = -1

        # add to data
        for i in range(len(states)):
            data.append((states[i], moves[i], winner))

    random.shuffle(data)
    X, y, win = zip(*data)
    return X, y, win
