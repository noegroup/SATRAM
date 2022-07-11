import math
from .tram import TRAM
from .satram import SATRAM
from .mbar import MBAR
from .sambar import SAMBAR


def get_solver(solver_type):
    if solver_type == "TRAM":
        return TRAM
    if solver_type == "SATRAM":
        return SATRAM
    if solver_type == "MBAR":
        return MBAR
    if solver_type == "SAMBAR":
        return SAMBAR


class ImplementationManager:

    def __init__(self, solver_type, lr_scheduler, initial_batch_size, patience, total_dataset_size):
        self.solver_type = solver_type
        self._solver = None

        self.patience = patience
        self.total_dataset_size = total_dataset_size

        if self.is_stochastic:
            self.batch_size = initial_batch_size
        else:
            self.batch_size = self.total_dataset_size

        self.lr_scheduler = lr_scheduler
        self._compute_learning_rate()

        self.set_solver()


    @property
    def solver(self):
        if self._solver is None:
            self.set_solver()
        return self._solver


    @property
    def is_stochastic(self):
        return self.solver_type == 'SATRAM' or self.solver_type == "SAMBAR"


    def set_solver(self):
        self._solver = get_solver(self.solver_type)


    def step(self, iteration, error=0):
        """Update the scheduler. The learning rate scheduler is updated first.
        Then, if the batch sizes is increased the learning rate doubles as well
        to match the new batch size. Returns True if the batch size was updated.
        """
        if self.is_stochastic:
            lr_stepped = self._lr_step(iteration, error)
            batch_size_stepped = self._batch_size_step(iteration)

            if lr_stepped or batch_size_stepped:
                self._compute_learning_rate()
                print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")

            return batch_size_stepped
        return False


    def _lr_step(self, iteration, error):
        if self.lr_scheduler is not None:
            return self.lr_scheduler.step(iteration, error)


    def _batch_size_step(self, iteration):
        if self.patience is not None:
            if iteration > 0 and iteration % self.patience == 0:
                self._increase_batch_size()

                if self.batch_size >= self.total_dataset_size:
                    self._switch_to_deterministic_implementation()

                # we have done an update on the implementation.
                return True

    def _compute_learning_rate(self):
        if self.batch_size > self.total_dataset_size:
            return 1
        lr = math.sqrt(self.batch_size / self.total_dataset_size)
        if self.lr_scheduler is not None:
            lr *= self.lr_scheduler.lr
        self.learning_rate = lr


    def _increase_batch_size(self):
        self.batch_size *= 2
        if self.lr_scheduler is not None:
            self.lr_scheduler.lr *= 2


    def _switch_to_deterministic_implementation(self):
        self.batch_size = self.total_dataset_size

        # switch to deterministic mode
        if self.solver_type == "SATRAM":
            self.solver_type = "TRAM"
        if self.solver_type == "SAMBAR":
            self.solver_type = "MBAR"

        self.set_solver()
