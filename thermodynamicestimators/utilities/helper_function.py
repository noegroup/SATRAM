import torch

def to_high_precision_tensor(data):
    if torch.is_tensor(data):
        return data.type(torch.float64)
    else:
        return torch.tensor(data, dtype=torch.float64)


def to_long_tensor(data):
    if torch.is_tensor(data):
        return data.type(torch.long)
    else:
        return torch.tensor(data, dtype=torch.long)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def ravel_index(indices, shape):
    out = torch.zeros(indices.shape[0])
    increment = 1
    for i in reversed(range(len(shape))):
        out += indices[:, i] * increment
        increment *= shape[i]
    return out

def to_histogram(samples, shape):
    """ Takes input list of samples and returns one histogram with the counts
    per bin summed over all states.

    Parameters
    ----------
    samples : torch.Tensor
        Tensor of shape (N, D) Where N is the number of samples
        and D is the dimensionality of the coordinates.
    shape : tuple
        Shape of the resulting histogram

    Returns
    -------
    N_per_bin : torch.Tensor
        Histogram of shape (shape) of the input samples.

    """

    # if the coordinates are 1d, get rid of the last dimension (of size 1)
    samples = samples.squeeze(-1)

    # make a histogram
    N_per_bin = torch.zeros(shape)

    # if more than 1-dimensional
    if len(shape) > 1:
        # flatten indices to 1D so we can use torch bincount
        samples = ravel_index(samples, shape).int()

    N_per_bin = torch.bincount(samples.type(torch.int), minlength=N_per_bin.numel())

    # if originally multi-dimensional: restore dimensions
    if len(shape) > 1:
        N_per_bin = N_per_bin.reshape(shape)

    return N_per_bin