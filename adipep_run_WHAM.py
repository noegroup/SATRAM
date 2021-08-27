import os.path
import torch
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from thermodynamicestimators.estimators import wham
import cProfile
from simtk import unit

ground_truth_path = "../tests/WHAM/peptide/input/ground_truth.txt"
discretized_samples_path = "../tests/WHAM/peptide/input/discretized_angles.h5"
bias_coefficients_path = "../tests/WHAM/peptide/input/bias_coefficients.txt"
angles_path = "../tests/MBAR/peptide/data_own/angles/angles_{}_{}.txt"
output_path = "../tests/WHAM/peptide/output/"
K = 200
n_bins = 50
n_therm_states = 625
n_bias_angles = 25


temperature = 300 * unit.kelvin
kT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temperature
kT_J = kT.value_in_unit(unit.kilojoules_per_mole)

kT_kcal = kT.value_in_unit(unit.kilocalories_per_mole)

def get_torsion_potential(theta_0, theta):

    diff = abs(theta-theta_0)
    dphi = min(diff, 2 * math.pi - diff)
    return 0.5 * K * dphi**2


def get_bias_coefficients():
    bias_coefficients = torch.zeros((n_therm_states, n_bins * n_bins), dtype=torch.float64)

    # thermodynamic states are defined by bias angles in two dimensions
    bias_angles = np.linspace(-math.pi, math.pi, n_bias_angles, endpoint=False)

    bin_angles = np.linspace(-math.pi, math.pi, n_bins, endpoint=False)

    # for every thermodynamic state (= phi_k X psi_k) get the angle for every bin
    for k in range(n_therm_states):
        i_k, j_k = divmod(k, n_bias_angles)
        phi_0 = bias_angles[i_k]
        psi_0 = bias_angles[j_k]

        for i_b, phi in enumerate(bin_angles):
            for j_b, psi in enumerate(bin_angles):
                bias = get_torsion_potential(phi_0, phi) + get_torsion_potential(psi_0, psi)
                bias_coefficients[k, n_bins * i_b + j_b] = -bias

    # force constant K is in kJ/mol so reduce bias coefficients to kT units
    return bias_coefficients / kT_J


def get_binned_samples():
    # samples are a list of flattened coordinates (between 0 and 624)
    binned_samples = torch.zeros((625000, 2), dtype=torch.long)

    # for every thermodynamic state (= phi_k X psi_k) get the angle for every bin
    for k in range(n_therm_states):
        print(k)
        i_k, j_k = divmod(k, n_bias_angles)

        # ... and get all samples and bin them accordingly
        angles_k = np.loadtxt(angles_path.format(i_k, j_k))
        # move range from (-180, 180) to [0,25)
        angles_k += 180
        angles_k /= 360
        angles_k *= n_bins
        angles_k = torch.Tensor(angles_k).type(torch.LongTensor).T
        binned_samples[k * 1000: (k+1) * 1000] = torch.stack((torch.ones(1000) * k, angles_k[0] * n_bins + angles_k[1])).T

    return binned_samples


def make_adipep_dataset():

    # gecheckt! Deze zijn goed
    if os.path.isfile(bias_coefficients_path):
        bias_coefficients = torch.Tensor(np.loadtxt(bias_coefficients_path))
    else:
        bias_coefficients = get_bias_coefficients()
        np.savetxt(bias_coefficients_path, bias_coefficients)


    if os.path.isfile(discretized_samples_path):
        with h5py.File(discretized_samples_path, 'r') as h5f:
            binned_samples = torch.Tensor(np.asarray(h5f['data'])).type(torch.LongTensor)
    else:
        binned_samples = get_binned_samples()
        with h5py.File(discretized_samples_path, 'w') as h5f:
            h5f.create_dataset('data', data=binned_samples)

    N_i = torch.histc(binned_samples[:,0].type(torch.FloatTensor), bins=n_therm_states)
    M_b = torch.histc(binned_samples[:,1].type(torch.FloatTensor), bins=n_bins * n_bins)

    return N_i, M_b, bias_coefficients, binned_samples


def plot_free_energies(data):
    result = torch.zeros((n_bias_angles, n_bias_angles))
    for k in range(n_therm_states):
        i, j = divmod(k, n_bias_angles)
        result[i,j] = data[k]

    vmin = torch.min(result)
    vmax = torch.max(result)
    fig, (ax1) = plt.subplots(1)
    im = ax1.contourf(result, vmin=vmin, vmax=vmax, levels=100, cmap='jet')
    plt.colorbar(im)
    plt.show()


def main():
    # generate a test problem with potential, biases, data and histogram bin range
    N_i, M_l, bias_coefficients, samples = make_adipep_dataset()

    dataloader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=True)

    if os.path.isfile(ground_truth_path):
        # in reduced kT units
        ground_truth = torch.Tensor(np.loadtxt(ground_truth_path))

    else:
        sci_estimator = wham.WHAM(N_i, M_l, bias_coefficients)#, log_file= "../tests/WHAM/peptide/output/sci_errors.txt")
        sci_estimator.estimate(dataloader, direct_iterate=True, max_iterations=100000, log_interval=10)#, ground_truth=ground_truth)
        ground_truth = sci_estimator.free_energies
        np.savetxt(ground_truth_path, sci_estimator.free_energies)

    # plot_free_energies(ground_truth)

    stoch_estimator = wham.WHAM(N_i, M_l, bias_coefficients, log_file= "../tests/WHAM/peptide/output/sgd_errors_0.txt")

    optimizer=torch.optim.Adam(stoch_estimator.parameters(), lr=1)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=600, verbose=True)

    # cp = cProfile.Profile()
    # cp.enable()

    stoch_estimator.estimate(dataloader, optimizer, schedulers=[lr_scheduler], max_iterations=10, ground_truth=ground_truth, tolerance=1e-3)#, log_interval=100)

    # cp.disable()
    # cp.print_stats()

    plot_free_energies(stoch_estimator.free_energies)

    # potential_stoch = stoch_estimator.get_potential(dataloader.dataset)



if __name__ == "__main__":
    main()