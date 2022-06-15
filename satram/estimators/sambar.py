import math
import torch.nn.functional as F
from ._common import *


def _compute_delta_f(f, N_k_log, normalized_N_k, bias, batch_size):
    return torch.exp(torch.logsumexp(F.log_softmax(N_k_log - bias + f, 1), axis=0)
                     - math.log(batch_size)) - normalized_N_k


def _update_f(f, N_k_log, normalized_N_k, bias, lr, batch_size):
    delta_f = _compute_delta_f(f, N_k_log, normalized_N_k, bias, batch_size)

    return f - lr * delta_f


def SAMBAR(dataset, f, log_v, lr, batch_size, *args, **kwargs):
    # allow for two-dimensional free energies in case we want to do MBAR iterations as an initialization for TRAM.
    if f.ndim == 2:
        f_therm = compute_f_therm(f)
    else:
        f_therm = f

    for batch_idx, batch_data in enumerate(dataset.dataloader):
        batch_data = batch_data.to(dataset.device)
        f_therm = _update_f(f_therm, dataset.N_k_log, dataset.normalized_N_k, batch_data[:, :dataset.n_therm_states],
                            lr, batch_size)

    if f.ndim == 2:
        f *= 0
        f += f_therm[:, None]
    else:
        f = f_therm
    return f, log_v
