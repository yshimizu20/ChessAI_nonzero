import torch
import torch.nn as nn

from chess_engine.model.model import ChessModel
from chess_engine.utils.dataloader import DataLoaderCluster, TestLoader

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.01
# batch_size = 200
# num_games = 200

cluster = DataLoaderCluster()
testing_iterator = TestLoader(
    "chess_engine/datasets/validation/lichess_elite_2023-06.pgn"
)


def train(
    start_epoch=0,
    end_epoch=5000,
    model_path=None,
):
    model = ChessModel()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    log_path = f"log_{start_epoch}.txt"

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)

    for epoch in range(start_epoch, end_epoch):
        X, y, win = cluster.get_data(100)

        X = torch.stack(X, dim=0).to(device)
        y = torch.stack(y, dim=0).to(device)
        win = torch.stack(win).unsqueeze(1).to(device)

        model.train()
        optimizer.zero_grad()

        policy, value = model(X)

        policy_loss = policy_criterion(policy, y)
        value_loss = value_criterion(value, win)

        total_loss = policy_loss + 0.75 * value_loss
        total_loss.backward()
        optimizer.step()

        scheduler.step()

        print(f"Epoch {epoch + 1} of {end_epoch}")
        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        with open(log_path, "a") as fp:
            fp.write(f"Epoch {epoch + 1} of {end_epoch}\n")
            fp.write(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}\n")

        del X, y, win, policy, value

        if epoch % 10 == 0:
            # evaluate
            X, y, win = (
                testing_iterator.X.to(device),
                testing_iterator.y.to(device),
                testing_iterator.win.to(device),
            )

            model.eval()
            with torch.no_grad():
                policy, value = model(X)
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
            torch.save(model.state_dict(), f"saved_models/model_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
