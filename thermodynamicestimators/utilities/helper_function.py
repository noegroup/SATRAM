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

