import numpy as np


def run(args, data, biases, beta=1.0, method='iterative'):

    total_sims = args.n_biases * args.n_simulations

    # bias coefficient matrix:
    c = np.zeros((total_sims, args.n_hist_bins))

    for i in range(total_sims):
        for j in range(args.n_hist_bins):
            c[i,j] = np.exp(-beta*biases[int(i/args.n_simulations)](j))

    if method == 'iterative':
        # Solve equations iteratively until self consistent:
        P = solve_iteratively(data, c, args)
    if method == 'SGD':
        P = solve_SGD(data, c, args)
    if method == 'ADAM':
        P = 0

    return -np.log(P)


def solve_iteratively(data, c, args):
    error = 1

    hist = np.histogram(data, range=(args.hist_min, args.hist_max), bins=args.n_hist_bins)[0]

    P = np.ones(args.n_hist_bins)  # unbiased probability per bin
    F = np.ones(args.n_simulations * args.n_biases)  # normalization factor per simulation

    epoch = 0
    # while not converged
    while error > args.tolerance and epoch < args.max_iterations:
        epoch += 1
        P_old = P

        # solve WHAM equations (this is what it's all about)
        P = hist / (len(data) * np.sum(F * c.T, axis=1))
        F = 1/np.sum(c * P, axis=1)

        # compute relative mean square error of the probabilities
        error = (np.square(np.subtract(P_old, P)).mean()) / P.mean()

    print("Iterative method done after {} epochs.".format(epoch))

    return P


def solve_SGD(data, c, args):
    error=1
    a = 0.01 # learning rate
    batch_size = 100
    n_batches = int(len(data)/batch_size)

    F = np.ones(args.n_simulations * args.n_biases)  # normalization factor per simulation
    P = np.ones(args.n_hist_bins)   # unbiased probability per bin

    epoch = 0
    while error > args.tolerance and epoch < args.max_iterations:
        epoch += 1

        np.random.shuffle(data)
        P_old = P

        for b in range(n_batches):

            hist = np.histogram(data[b*batch_size:(b+1)*batch_size], range=(0, 100), bins=args.n_hist_bins)[0]

            P = (1-a) * P + a * hist / (batch_size * np.sum(F * c.T, axis=1))
            F = (1-a) * F + a * 1/np.sum(c * P, axis=1)

        error = np.square(np.subtract(P, P_old)).mean() / P.mean()

    print("SGD done after {} epochs.".format(epoch))
    return P
