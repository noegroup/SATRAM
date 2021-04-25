import numpy as np
import pymbar
import thermodynamicestimators.utilities.test_problem_factory as problem_factory
import matplotlib.pyplot as plt

def run():
    test_problem = problem_factory.make_test_problem('double_well_1D')

    u_kn = test_problem.data_at_all_states.numpy()
    N_k = np.asarray([1000] * len(u_kn))
    mbar = pymbar.MBAR(u_kn, N_k)


    # plt.plot(mbar.getFreeEnergyDifferences()[0])
    plt.plot(mbar.f_k)
    plt.show()


if __name__ == '__main__':
    run()