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


class ImplementationManager():

    def __init__(self, solver_type, initial_batch_size, batch_size_increase, total_dataset_size):
        self._solver = None

        self.current_batch_size = initial_batch_size
        self.batch_size_increase_interval = batch_size_increase
        self.total_dataset_size = total_dataset_size

        # TODO: compute this based on data size en available memory
        self.batch_size_memory_limit = 8192
        self.batch_size = initial_batch_size

        self.learning_rate = self._compute_learning_rate()

        self.solver_type = solver_type

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


    def step(self, iteration):
        if self.batch_size_increase_interval is not None:
            if self.solver_type == 'SATRAM' or self.solver_type == "SAMBAR":

                if iteration % self.batch_size_increase_interval == 0:
                    self._increase_batch_size()

                    if self.batch_size >= self.total_dataset_size:
                        self._switch_to_deterministic_implementation()

                    # we have done an update on the implementation.
                    print(f"increasing batch size to {self.batch_size}, lr to {self.learning_rate}")
                    return True
        return False


    def _compute_learning_rate(self):
        if self.batch_size > self.total_dataset_size:
            return 1
        return math.sqrt(self.batch_size / self.total_dataset_size)


    def _increase_batch_size(self):
        self.batch_size *= 2
        self.learning_rate = self._compute_learning_rate()


    def _switch_to_deterministic_implementation(self):
        self.batch_size = self.total_dataset_size

        # switch to deterministic mode
        if self.solver_type == "SATRAM":
            self.solver_type = "TRAM"
        if self.solver_type == "SAMBAR":
            self.solver_type = "MBAR"

        self.set_solver()
