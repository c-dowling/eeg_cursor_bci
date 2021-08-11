import torch


class RandomWindow:
    """Crops signal randomly. """
    def __init__(self, width):
        self.width = width

    def __call__(self, x):
        start = torch.randint(low=0, high=x.shape[1] - self.width, size=(1, ))
        return x[:, start:start + self.width]
