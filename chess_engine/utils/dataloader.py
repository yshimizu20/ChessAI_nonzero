import torch

from chess_engine.utils.state import createData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, "r")

    def _create_data(self, step):
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

    def get_data(self, step=250):
        return self._create_data(step)


class TestLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y, self.w = self.get_data(250)
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)
        self.fp.close()
