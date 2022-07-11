import abc


class Scheduler:
    def __init__(self, initial_lr=1e-3):
        self.lr = initial_lr

    @abc.abstractmethod
    def step(self, *args):
        pass


class GammaScheduler(Scheduler):
    def __init__(self, initial_lr, patience=1, gamma=1., minimum_lr=0.):
        super().__init__(initial_lr)
        self.patience = patience
        self.gamma = gamma
        self.i = 0
        self.minimum_lr = minimum_lr

    def step(self, *args):
        if self.lr > self.minimum_lr:
            self.i += 1
            if self.i == self.patience:
                self.lr *= self.gamma
                self.i = 0
                return True
