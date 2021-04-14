"""
MCMC.py

Monte Carlo Markov Chain sampler for sampling a trajectory in d dimensions.
"""

import numpy as np
from numpy.random import randint, uniform


class MCMC:
    """
    Initialize the MCMC sampler

    Parameters
    ----------
    args : object
        The input arguments of the program

    Returns
    -------
    MCMC : MCMC
        the MCMC sampler
    """
    def __init__(self, args):
        self.x_min = args.hist_min
        self.x_max = args.hist_max + 1
        self.max_step = 2
        self.d = args.n_dimensions
        self.n_steps = args.n_samples
        self.n_simulations = args.n_simulations


    """
    Given a potential and a list of biases, sample #n_simulations trajectories per bias. Trajectories are
    returns as one np array.

    Parameters
    ----------
    U : function
        The potential function. When given coordinates, returns value of the potential at those coordinated
    biases : list of functions
        The biases that are added to the potential. For each bias, #n_simulations simulations are run.
    n_simulations: int
        Number of simulations to run for each bias.

    Returns
    -------
    results : np array
        np array of all coordinates of all sampled trajectories
    """
    def sample(self, U, biases):
        results = []

        for bias in biases:
            biased_potential = lambda x: U(x) + bias(x)

            for s in range(self.n_simulations):
                samples = self.get_trajectory(biased_potential)
                results.append(samples)

        return np.asarray(results).flatten()

    """
    Sample a trajectory using Markov Chain Monte Carlo
    
    Parameters
    ----------
    U : function
        The potential function. When given coordinates, returns value of the potential at those coordinated
    beta : float, Optional, default = 1.0
        Inverse temperature
    n_steps: int, Optional, default = 10000
        Size of the returned trajectory 

    Returns
    -------
    trajectory : np array
        np array of length n_steps containing the coordinates of the sampled trajectory
    """
    def get_trajectory(self, U, beta=1.0):
        p = lambda u: np.exp(-beta*u)

        r_prev = np.random.randint(self.x_min, self.x_max, self.d)

        trajectory = []

        steprange = (-self.max_step, self.max_step + 1)

        for n in range(self.n_steps):

            r = r_prev + randint(steprange[0], steprange[1], self.d)
            while (r < self.x_min).any() or (r > self.x_max).any():
                r = r_prev + randint(steprange[0], steprange[1], self.d)

            delta = U(r) - U(r_prev)
            if delta > 0:
                # print p(delta)
                if p(delta) < uniform(0,1):
                    r = r_prev
                else:
                    r_prev = r
            else:
                r_prev = r

            trajectory.append(r)
        return np.asarray(trajectory)