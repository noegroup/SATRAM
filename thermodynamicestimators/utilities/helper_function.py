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