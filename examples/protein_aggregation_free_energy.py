import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from tqdm.notebook import tqdm
from satram import ThermodynamicEstimator
from deeptime.markov.msm import TRAM, TRAMDataset


print(f"Is cuda available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

print(f"Cuda device name: {torch.cuda.get_device_name(0)}")

T_sim = 310.0
T_st = 298.0

# Energy observables in the simulation code are in units of standard kT/particle
en_factor = T_st / T_sim

# The membrane patch is originally a lattice of 50x58 particles
nx = 50
ny = 58

# Number of particles
n_particles = 2 * nx * ny

# Different stiffness values of simulated peripheral proteins (50, 100, and 200 MPa)
protein_stiffness_list = [50, 100, 200]

burn_in_steps = 2167

q_table = []
u_red_table = []
conc_table = []
time_table = []

for stiffness in protein_stiffness_list:
    file_name = "datasets/time_series_stxb_stiffness_" + str(stiffness) + ".npz"

    q, u_red, conc, time = (np.load(file_name)['arr_' + str(i)] for i in range(4))

    q_table.append(q[:, burn_in_steps:].copy() * 1.0e3)
    u_red_table.append(u_red[:, burn_in_steps:].copy() * n_particles * en_factor)
    conc_table.append(conc.copy())
    time_table.append(time[:, burn_in_steps:].copy())

q_table = np.array(q_table)
u_red_table = np.array(u_red_table)
conc_table = np.array(conc_table)
time_table = np.array(time_table)

n_bins = 150

q_min, q_max = np.amin(q_table), np.amax(q_table)

q_bins = np.linspace(0.9999 * q_min, 1.0001 * q_max, n_bins + 1)

q_bins_center = 0.5 * (q_bins[:-1] + q_bins[1:])

q_discrete_table = []
q_hist_table = []

# fig = plt.figure(figsize=(15, 6))
# fig.subplots_adjust(wspace=0.3)

for i, (qq, tt, _stiffness) in enumerate(zip(q_table, time_table, protein_stiffness_list)):

    q_discrete = []
    q_hist = []

    # ax_top = fig.add_subplot(2, 3, i + 1)
    # ax_btm = fig.add_subplot(2, 3, i + 4)

    # ax_top.set_title(r"$Y_\mathrm{p}=$" + "{} MPa".format(_stiffness))

    for q, t in zip(qq, tt):
        # ax_top.plot(t, q, alpha=0.6)

        hist, bin_edges = np.histogram(q, bins=q_bins)

        # ax_btm.plot(q_bins_center, hist / np.trapz(hist, q_bins_center))

        q_discrete.append(np.digitize(q, bins=q_bins, right=True))
        q_hist.append(hist.copy())

    q_discrete_table.append(q_discrete.copy())
    q_hist_table.append(q_hist.copy())

    # ax_top.set_xlabel("time [ms]")
    # ax_top.set_ylabel(r"q [$\mu\mathrm{m}^{-1}$]")

    # ax_btm.set_xlabel(r"q [$\mu\mathrm{m}^{-1}$]")
    # ax_btm.set_ylabel("pdf [$\mu\mathrm{m}$]")

q_discrete_table = np.array(q_discrete_table)
q_hist_table = np.array(q_hist_table)

lag_time = 500


def u_function(q_, a):
    result = np.zeros_like(q_)

    qq = np.ones_like(q_)

    for _a in a:
        result += _a * qq

        qq = qq * q_.copy()

    return result


u_mean_0_table = []
energy_func_xi_table = []
energy_func_coeff_table = []

for stiffness in protein_stiffness_list:
    file_name = "datasets/bias_stxb_stiffness_" + str(stiffness) + ".npz"

    u_mean_0, energy_func_xi, energy_func_coeff = (np.load(file_name)['arr_' + str(i)] for i in range(3))

    u_mean_0_table.append(u_mean_0.copy())
    energy_func_xi_table.append(energy_func_xi.copy())
    energy_func_coeff_table.append(energy_func_coeff.copy())

u_mean_0_table = np.array(u_mean_0_table)
energy_func_xi_table = np.array(energy_func_xi_table)
energy_func_coeff_table = np.array(energy_func_coeff_table)


def convert_to_tensor(x_table):
    x_tensor = []

    for x in x_table:
        x_tensor.append(torch.from_numpy(x))

    return x_tensor


fs = []


def callback(f, log_v):
    fs.append(f)


max_iter = 20

free_energy_table = []
pmf_table = []
weights_table = []

for q_discrete, \
    u_red, u_mean_0, \
    energy_func_xi, energy_func_coeff in zip(q_discrete_table,
                                             u_red_table, u_mean_0_table,
                                             energy_func_xi_table,
                                             energy_func_coeff_table):

    n_therm_states, n_samples = q_discrete.shape
    ttrajs = []

    for i in range(n_therm_states):
        ttrajs.append(np.ones(n_samples, dtype=np.int32) * i)

    # ttrajs = convert_to_tensor(ttrajs)
    dtrajs = q_discrete

    free_energy_sample = []
    pmf_sample = []
    weights_sample = []

    # For each case of protein stiffness and concentration, there are 5 different inferred bias potentials
    for _xi, _coeff in zip(energy_func_xi, energy_func_coeff):

        bias_matrices = np.zeros((n_therm_states, n_samples, n_therm_states))

        for i, (_q_discrete, _u_red) in enumerate(zip(q_discrete, u_red)):

            _q = q_bins[_q_discrete]

            for k in range(n_therm_states):
                _bias_func = lambda __q: u_mean_0 + _xi[k] * u_function(__q, _coeff) - _u_red

                bias_matrices[i, :, k] = _bias_func(_q)

        # bias_matrices = convert_to_tensor(bias_matrices)

        estimator = ThermodynamicEstimator(lagtime=lag_time, progress=tqdm,
                                           maxerr=1.0e-5, maxiter=500, device='cuda')

        estimator.fit((ttrajs, dtrajs, bias_matrices),
                      solver_type="SATRAM", callback=callback, patience=4)
        # plt.show()
        dataset = TRAMDataset(ttrajs=ttrajs, dtrajs=dtrajs, bias_matrices=bias_matrices, lagtime=lag_time)
        deeptime_TRAM = TRAM(progress=tqdm).fit_fetch(dataset)
        pmf = deeptime_TRAM.compute_PMF(dtrajs, bias_matrices, q_discrete, therm_state=1)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # im = ax1.contourf(deeptime_TRAM.biased_conf_energies - deeptime_TRAM.biased_conf_energies[0,1], levels=50)
        # ax2.contourf(estimator._f.cpu().numpy() - estimator._f.cpu().numpy()[0,1], levels=im.levels)
        # plt.show()

        weights = deeptime_TRAM.compute_sample_weights_log(dtrajs, bias_matrices)
        free_energy_sample.append(estimator.free_energies_per_thermodynamic_state.numpy())
        pmf_sample.append(estimator.compute_pmf(torch.Tensor(q_discrete), n_bins, therm_state=1).numpy())
        plt.plot(pmf)
        plt.plot(pmf_sample[-1])
        plt.show()
        weights_sample.append(estimator.sample_weights().cpu().numpy())

        #         plt.matshow(np.mean(bias_matrices, axis=1))

        if np.any(np.isnan(pmf_sample)):
            raise ValueError("NaN detected!")

    free_energy_table.append(free_energy_sample.copy())
    pmf_table.append(pmf_sample.copy())
    weights_table.append(weights_sample.copy())
