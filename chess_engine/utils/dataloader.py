import torch
import time

from chess_engine.utils.state import createData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, "r")

    def _create_data(self, step=200):
        X, y, win = createData(self.fp, step)
        if len(win) < step:
            self.fp.close()
            self.fp = open(self.filename, "r")
            print("reached end of", self.filename)
            with open("log_training.txt", "a") as tfp:
                tfp.write(f"reached end of {self.filename}\n")

        return X, y, win

    def get_data(self, step=200):
        return self._create_data(step)


class DataLoaderCluster:
    def __init__(self):
        self.loaders = [
            DataLoader(f"filtered/output-{yr}_{mo:02d}.pgn")
            for mo in range(1, 13)
            for yr in range(2015, 2018)
        ]
        self.idx = 0

    def get_data(self, step=50):
        X, y, win = self.loaders[self.idx].get_data(step)
        if len(win) < step:
            self.idx += 1
            if self.idx == len(self.loaders):
                self.idx = 0
            new_X, new_y, new_win = self.loaders[self.idx].get_data(step - len(win))
            X.extend(new_X)
            y.extend(new_y)
            win.extend(new_win)

        return X, y, win


class TestLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        X, y, win = self.get_data(100)
        self.X = torch.stack(X, dim=0).to(device)
        self.y = torch.stack(y, dim=0).to(device)
        self.win = torch.stack(win).unsqueeze(1).to(device)
        self.fp.close()
