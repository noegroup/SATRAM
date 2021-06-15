import torch
import numpy as np
from thermodynamicestimators.data_sets.free_energy_dataset import FreeEnergyDataset
from thermodynamicestimators.estimators.mbar import MBAR
import tables
import argparse
from simtk import unit

temperature = 300 * unit.kelvin
kT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temperature
kT = kT.value_in_unit(unit.kilocalories_per_mole)


input_file_path = "C:\\Users\\Maaike\\Documents\\alanine_dipeptide\\ala2_shuffled.h5"
ground_truth_file_path = "C:\\Users\\Maaike\\Documents\\alanine_dipeptide\\pymbar_F_adaptive_{}.txt"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int)
    args = parser.parse_args()
    i = args.i
    N_i = np.ones((625)) * 100 * i

    with tables.open_file(input_file_path, 'r', driver="H5FD_CORE", driver_core_backing_store=0) as h5_file:
        data = np.asarray(h5_file.root['data'])

    data_size = int(625 * i * 100)
    data_i = data[:data_size] * kT

    dataset = FreeEnergyDataset(torch.Tensor(data_i), torch.Tensor(N_i))
    # dataset = H5Dataset(filepath, N_i=torch.Tensor(N_i))

    ground_truth = np.loadtxt(ground_truth_file_path.format(i)) * kT

    torch.random.manual_seed(1234)

    slowmbar = MBAR(n_states=625)

    optimizer = torch.optim.Adam(slowmbar.parameters(), lr=1 )

    batch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, threshold=0.0001,
                                            threshold_mode='rel', cooldown=0, min_lr=0.005, eps=1e-08, verbose=True)

    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0,
                                                                 threshold=1e-06,
                                                                 threshold_mode='rel', cooldown=0, min_lr=0.00000001,
                                                                 eps=1e-08, verbose=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8,
                                             prefetch_factor=1024)

    log_file_name = f'Stoch_MBAR_log_{i}.txt'
    free_energy_file_name = f'Stoch_MBAR_F_per_iteration_{i}.txt'

    free_energies, errors = slowmbar.estimate(dataloader, dataset, optimizer, epoch_scheduler=epoch_scheduler, batch_scheduler=batch_scheduler,
                                              tolerance=1e-1, ground_truth=ground_truth, max_iterations=100, log_interval=50)

