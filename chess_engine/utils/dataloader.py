import torch

from chess_engine.utils.state import createData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, "r")

    def _create_data(self, step=200):
        try:
            X, y, win = createData(self.fp, step)
        except StopIteration:
            self.fp.close()
            self.fp = open(self.filename, "r")
            print("reached end of", self.filename)
            try:
                with open("log_training.txt", "a") as tfp:
                    tfp.write(f"reached end of {self.filename}\n")
            except:
                pass
            raise StopIteration

        return X, y, win

    def get_data(self, step=200):
        return self._create_data(step)


class TestLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        X, y, win = self.get_data(100)
        self.X = torch.stack(X, dim=0).to(device)
        self.y = torch.stack(y, dim=0).to(device)
        self.win = torch.stack(win).unsqueeze(1).to(device)
        self.fp.close()
