#!/usr/bin/env python
#SBATCH -J stoch_mbar
#SBATCH -D /data/scratch/galam92/adipep/output/data_efficiency/gpu
#SBATCH -o stoch_mbar.%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=10M
#SBATCH --time=32:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=galam92@zedat.fu-berlin.de

import numpy as np
import torch
from thermodynamicestimators.data_sets.free_energy_dataset import FreeEnergyDataset
from thermodynamicestimators.estimators.mbar import MBAR
import tables
import argparse
import os
import time
import platform

torch.multiprocessing.set_sharing_strategy('file_system')

kT = 0.5961612775922495
#temperature = 300 * unit.kelvin
#kT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temperature
#kT = kT.value_in_unit(unit.kilocalories_per_mole)

input_file_path = "/data/scratch/galam92/adipep/output/ala2_shuffled.h5"
ground_truth_file_path = "/data/scratch/galam92/adipep/output/data_efficiency/pymbar_F_adaptive_{}.txt"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int)
    parser.add_argument("-r", type=int)
    args = parser.parse_args()
    i = args.i
    r = args.r
    N_i = np.ones((625)) * 100 * i
    N_i = torch.Tensor(N_i)

    print(f"i = {i} \n")
    print("pltform.processor: {}".format(platform.processor()))
    print("CUDA available: {}".format(torch.cuda.is_available()))

    free_energy_file_name = f'/data/scratch/galam92/adipep/output/data_efficiency/gpu/Stoch_MBAR_GPU_F_per_iteration_{i}_{r}.txt'

    with tables.open_file(input_file_path, 'r', driver="H5FD_CORE", driver_core_backing_store=0) as h5_file:
        data = np.asarray(h5_file.root['data'])

    data_size = int(625 * i * 100)
    data_i = (torch.Tensor(data[:data_size]) * kT)

    t0 = time.time()
    dataset = FreeEnergyDataset(data_i, N_i)

    ground_truth = torch.Tensor(np.loadtxt(ground_truth_file_path.format(i)) * kT)

    torch.random.manual_seed(1234 + r)
    
    slowmbar = MBAR(n_states=625, free_energy_log=free_energy_file_name, device="cuda")

    optimizer = torch.optim.Adam(slowmbar.parameters(), lr=1 )

    batch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, threshold=0.0001,
                                            threshold_mode='rel', cooldown=0, min_lr=0.005, eps=1e-08, verbose=True)

    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0,
                                                                 threshold=1e-06,
                                                                 threshold_mode='rel', cooldown=0, min_lr=0.00000001,
                                                                 eps=1e-08, verbose=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=1, shuffle=True, prefetch_factor=1024)


    converged = slowmbar.estimate(dataloader, dataset, optimizer, epoch_scheduler=epoch_scheduler, batch_scheduler=batch_scheduler,
                                              tolerance=1e-1, ground_truth=ground_truth, max_iterations=100, log_interval=50)

    t1=time.time()
    running_time = t1 - t0
    timeString = time.strftime("%H:%M:%S", time.gmtime(running_time))
    with open(f"stoch_MBAR_GPU_running_times_{i}.txt", "a") as f:
        f.write(f"{r} {converged} {timeString}\n")
