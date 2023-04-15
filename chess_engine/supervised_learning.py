import torch
import torch.nn as nn

from chess_engine.model.policy import PolicyNetwork
from chess_engine.model.value import ValueNetwork
from chess_engine.utils.dataloader import DataLoaderCluster, TestLoader

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
batch_size = 200
num_games = 200

cluster = DataLoaderCluster()
testing_iterator = TestLoader("filtered/db2023.pgn")


def train(
    start_epoch=0,
    end_epoch=5000,
    policy_model_path=None,
    value_model_path=None,
):
    policynet = PolicyNetwork()
    valuenet = ValueNetwork()
    if policy_model_path is not None:
        policynet.load_state_dict(torch.load(policy_model_path))
    if value_model_path is not None:
        valuenet.load_state_dict(torch.load(value_model_path))
    policynet.to(device)
    valuenet.to(device)

    num_epochs = end_epoch - start_epoch
    log_path = f"log_{start_epoch}.txt"

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        [{"params": policynet.parameters()}, {"params": valuenet.parameters()}], lr=lr
    )

    for epoch in range(start_epoch, end_epoch):
        X, y, win = cluster.get_data()

        X = torch.stack(X, dim=0).to(device)
        y = torch.stack(y, dim=0).to(device)
        win = torch.stack(win).unsqueeze(1).to(device)

        # train policy network
        policy_loss = 0
        policy = policynet(X)
        policy_loss = policy_criterion(policy, y)

        # train value network
        value_loss = 0
        value = valuenet(X)
        value_loss = value_criterion(value, win)

        # train both networks
        optimizer.zero_grad()

        alpha = 0.5
        beta = 0.5
        loss = alpha * policy_loss + beta * value_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {num_epochs}\n")
            fp.write(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        del X, y, win, policy, value

        if epoch % 10 == 0:
            # evaluate
            X, y, win = testing_iterator.X, testing_iterator.y, testing_iterator.win

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


if __name__ == "__main__":
    train()
