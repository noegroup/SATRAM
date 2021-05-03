import torch


class ThermodynamicEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    ''' The relative free energy estimate. One value for each thermodynamic state. '''

    @property
    def free_energy(self):
        return NotImplemented

    ''' The value of the objective function that needs to be minimized to obtain the free energies. '''

    def residue(self, data):
        return NotImplemented

    ''' Subtract the first free energy from all free energies such that the first is zero and all other energies are
    relative to the first. '''

    def shift_free_energies_relative_to_zero(self):
        return NotImplemented

    def estimate(self, data_loader, optimizer=None, scheduler=None, dataset=None, tolerance=1e-2, max_iterations=1000,
                 direct_iterate=False, ground_truth = None):

        epoch = 0
        error = tolerance + 1
        errors = []
        # free_energies = []

        while epoch < max_iterations and error > tolerance:

            epoch += 1

            for (idx, batch) in enumerate(data_loader):

                if direct_iterate:
                    self.self_consistent_step(batch)

                else:
                    optimizer.zero_grad()
                    loss = self.residue(batch)
                    loss.backward()
                    optimizer.step()


            if not scheduler is None:
                scheduler.step()

            # avoid free energies getting to large by shifting them back towards zero.
            self.shift_free_energies_relative_to_zero()

            error = torch.abs(torch.square(self.free_energy - ground_truth).mean() / ground_truth.mean())

            # free_energy_unbiased = -torch.log(self.get_unbiased_partition_function(dataset[:]))

            print(error)
            errors.append(error)
            # free_energies.append(free_energy_unbiased)

        return self.free_energy, errors
