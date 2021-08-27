import torch
import matplotlib.pyplot as plt
from thermodynamicestimators.test_cases import test_case_factory
from thermodynamicestimators.estimators import wham


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    test_problem_name = 'double_well_1D'

    # generate a test problem with potential, biases, data and histogram bin range
    test_case = test_case_factory.make_test_case(test_problem_name)
    N_i, M_l, bias_coefficients, samples = test_case.to_wham_dataset()

    dataloader = torch.utils.data.DataLoader(samples, batch_size=128, shuffle=True)

    estimator = wham.WHAM(N_i, M_l, bias_coefficients)

    optimizer=torch.optim.Adam(estimator.parameters(), lr=1)
    # optimizer = torch.optim.Adam(estimator.parameters(), lr=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5* len(dataloader), verbose=True)
    estimator.estimate(dataloader, optimizer, ground_truth=test_case.ground_truth, max_iterations=10000)#, schedulers=[lr_scheduler])
    potential_SGD = estimator.get_potential()

    # optimizer_ADAM = torch.optim.Adam(estimator.parameters(), lr=0.01)
    # free_energy_ADAM, errors_ADAM = estimator.estimate(dataloader, optimizer_ADAM)
    # potential_ADAM = estimator.get_potential(dataset[:])

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    # estimator.estimate(dataloader, dataset, direct_iterate=True, ground_truth=test_case.ground_truth)


    plt.plot(estimator.free_energies, label='SGD')
    # plt.plot(free_energy_sci, label='sci')
    plt.plot(test_case.ground_truth, label='Ground truth')
    plt.legend()
    plt.show()

    plt.yscale('log')
    plt.plot(estimator.errors, label='SGD error, lr=0.01')
    # plt.plot(errors_sci, label='SCI error')

    plt.legend()
    plt.show()


    if test_problem_name == "double_well_1D":
        plt.plot([test_case.potential_fn(x) for x in range(100)], label="real potential function", color='g')
        plt.plot(potential_SGD, label="SGD")
        # plt.plot(potential_ADAM, label="ADAM")

    if test_problem_name == "double_well_2D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = torch.tensor(range(test_case.histogram_range[0][0], test_case.histogram_range[0][1]))
        y = torch.tensor(range(test_case.histogram_range[1][0], test_case.histogram_range[1][1]))
        X, Y = torch.meshgrid(x, y)

        # real_potential = torch.zeros_like(X)
        # for r, _ in torch.ndenumerate(real_potential):
        #     real_potential[r] = dataset.potential((X[r], Y[r])))

        ax.plot_wireframe(X, Y, potential_SGD - potential_SGD[~torch.isinf(potential_SGD)].mean(), label="SGD", color='b')
        # ax.plot_wireframe(X, Y, potential_ADAM - potential_ADAM[~torch.isinf(potential_ADAM)].mean(), label="ADAM", color='r')
        # ax.plot_wireframe(X, Y, real_potential - real_potential.mean(), label="Real potential function", color='g')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()