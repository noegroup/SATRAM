"""
MCMC.py

Monte Carlo Markov Chain sampler for sampling a trajectory in d dimensions.
"""

import torch


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
    def __init__(self, sampling_range, max_step, n_dimensions, n_samples):
        self.sampling_range = sampling_range
        self.max_step = max_step
        self.d = n_dimensions
        self.n_steps = n_samples

        torch.manual_seed(1000)


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
    def get_trajectory(self, U, beta=1.0, r_initial=None):
        p = lambda u: torch.exp(-beta*u)

        r_prev=r_initial

        if r_prev is None:
            r_prev = torch.tensor(
                [self.get_step_from_uniform(self.sampling_range[d][0], self.sampling_range[d][1], 1) for d in range(self.d)]
            )

        trajectory = []

        for n in range(self.n_steps):

            r = r_prev + self.get_step_from_uniform(-self.max_step, self.max_step, self.d)

            while (r < self.sampling_range[:,0]).any() or (r >= self.sampling_range[:,1]).any():
                r = r_prev + self.get_step_from_uniform(-self.max_step, self.max_step, self.d)

            delta = U(r) - U(r_prev)
            if delta > 0:
                # print p(delta)
                if p(delta) < torch.rand(1).item():
                    r = r_prev
                else:
                    r_prev = r
            else:
                r_prev = r

            trajectory.append(r)
        return torch.stack(trajectory)


    def get_step_from_uniform(self, min, max, size):
        return (max - min) * torch.rand(size) + min