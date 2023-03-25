import torch
import torch.nn as nn

from chess_engine.model.policy import PolicyNetwork
from chess_engine.utils.dataloader import DataLoader


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
batch_size = 200

training_iterators = [
    DataLoader(f"data/filtered/output-{year}_{month:02d}.pgn")
    for year in range(2015, 2018)
    for month in range(1, 13)
]
training_idx = 0

testing_iterator = DataLoader("data/filtered/db2023.pgn")


def main(loadpath=None):
    policynet = PolicyNetwork()
    if loadpath:
        policynet.load_state_dict(torch.load(loadpath))

    policy_criterion = nn.CrossEntropyLoss()
    policy_optimizer = torch.optim.Adam(policynet.parameters(), lr=lr)

    train(policynet, policy_criterion, policy_optimizer)


def train(policynet, policy_criterion, policy_optimizer, num_epochs=5000):
    for epoch in range(num_epochs):
        with open("log20000.txt", "a") as f:
            f.write(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Epoch {epoch + 1} of {num_epochs}")
        running_loss = 0.0

        # Get the inputs
        try:
            X, y, _ = training_iterators[training_idx].get_data(200)
        except StopIteration:
            training_idx += 1
            if training_idx == len(training_iterators):
                training_idx = 0
            continue

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        # Zero the parameter gradients
        policy_optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = policynet(X)
        loss = policy_criterion(outputs, y)
        loss.backward()
        policy_optimizer.step()

        # Print statistics
        running_loss += loss.item()
        with open("log20000.txt", "a") as f:
            f.write("[%d] training loss: %.5f" % (epoch + 1, running_loss))
        print("[%d] training loss: %.5f" % (epoch + 1, running_loss))
        running_loss = 0.0

        del X
        del y
        del outputs

        X, y = testing_iterator.X, testing_iterator.y
        if epoch % 10 == 0:
            with torch.no_grad():
                outputs = policynet(X)
                loss = policy_criterion(outputs, y)
            running_loss += loss.item()
            with open("log20000.txt", "a") as f:
                f.write("[%d] test loss: %.5f" % (epoch + 1, running_loss))
            print("[%d] test loss: %.5f" % (epoch + 1, running_loss))
            running_loss = 0.0
            del X
            del y
            del outputs

        if epoch % 100 == 9:
            torch.save(policynet.state_dict(), f"models/model_{epoch}.pt")


if __name__ == "__main__":
    main()
