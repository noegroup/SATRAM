#!/usr/bin/env python
#SBATCH -J stoch_mbar
#SBATCH -D /data/scratch/galam92/adipep/output/batch_size_test/cpu
#SBATCH -o stoch_mbar.%j.out
#SBATCH --partition=small
#SBATCH --constraint="AMD"
#SBATCH --exclude=gpu[036-083]
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
ground_truth_file_path = "/data/scratch/galam92/adipep/output/batch_size_test/pymbar_F_adaptive.txt"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int)
    parser.add_argument("-r", type=int)
    args = parser.parse_args()
    batch_size = args.b
    r = args.r
    N_i = torch.Tensor(np.ones((625)) * 1000)

    print(f"batch size: {batch_size}. Run {r}. \n")
    print("pltform.processor: {}".format(platform.processor()))
    print("CUDA available: {}".format(torch.cuda.is_available()))

    free_energy_file_name = f'/data/scratch/galam92/adipep/output/batch_size_test/Stoch_MBAR_F_per_iteration_{batch_size}_{r}.txt'

    with tables.open_file(input_file_path, 'r', driver="H5FD_CORE", driver_core_backing_store=0) as h5_file:
        data = np.asarray(h5_file.root['data'])

    data_size = int(625 * 1000)
    data = torch.Tensor(data) * kT

    t0 = time.time()
    dataset = FreeEnergyDataset(data, N_i)

    ground_truth = torch.Tensor(np.loadtxt(ground_truth_file_path) * kT)

    torch.random.manual_seed(1234 + r)
    
    slowmbar = MBAR(n_states=625, free_energy_log=free_energy_file_name, device="cpu")

    optimizer = torch.optim.Adam(slowmbar.parameters(), lr=1 )

    batch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, threshold=0.0001,
                                            threshold_mode='rel', cooldown=0, min_lr=0.005, eps=1e-08, verbose=True)

    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0,
                                                                 threshold=1e-06,
                                                                 threshold_mode='rel', cooldown=0, min_lr=0.00000001,
                                                                 eps=1e-08, verbose=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True )

    log_interval = int(len(dataloader)/10) 
    if log_interval == 0:
        log_interval = int(len(dataloader)/4)
    if log_interval == 0:
        log_interval == int(len(dataloader)/2)
    if log_interval == 0:
        log_interval = 1

    converged = slowmbar.estimate(dataloader, dataset, optimizer, epoch_scheduler=epoch_scheduler, batch_scheduler=batch_scheduler,
                                              tolerance=1e-1, ground_truth=ground_truth, max_iterations=100, log_interval=log_interval)

    t1=time.time()
    running_time = t1 - t0
    timeString = time.strftime("%H:%M:%S", time.gmtime(running_time))
    with open(f"stoch_MBAR_running_times_{batch_size}.txt", "a") as f:
        f.write(f"{r} {converged} {timeString}\n")
