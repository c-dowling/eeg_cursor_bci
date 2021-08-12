import copy

import torch


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.current_patience = patience
        self.best_model = None
        self.best_loss = float('inf')

    def __call__(self, model, validation_loss, epoch):
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_model = copy.deepcopy(model)
            self.current_patience = self.patience
        else:
            self.current_patience -= 1

        if self.current_patience == 0:
            print(f'Training halted. Retrieving weights from epoch {epoch}. Best validation loss {self.best_loss:.4f}')
            model.load_state_dict(self.best_model.state_dict())
            return True
        return False
