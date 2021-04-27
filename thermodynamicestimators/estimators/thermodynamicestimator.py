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


    def estimate(self, data_loader, optimizer, tolerance=1e-3, max_iterations=1000):

        epoch = 0
        error = tolerance + 1
        free_energy = 0
        errors = []

        while epoch < max_iterations and error > tolerance:

            epoch += 1

            for (idx, batch) in enumerate(data_loader):

                optimizer.zero_grad()
                loss = self.residue(batch)
                loss.backward()
                optimizer.step()

            # avoid free energies getting to large by shifting them back towards zero.
            self.shift_free_energies_relative_to_zero()


            error = torch.max(torch.square(
                (self.free_energy - free_energy) / (0.1 * torch.abs(self.free_energy.mean()))))
            free_energy = self.free_energy

            print(error)
            errors.append(error)

        return self.free_energy, errors
