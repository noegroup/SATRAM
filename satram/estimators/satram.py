import math
import torch.nn.functional as F
from ._common import *



def _compute_batch_delta_f(f, log_R, bias, ind_trajs):
    return torch.logsumexp(F.log_softmax(log_R + f - bias[:, :, None], 1) + torch.log(ind_trajs[:, None, :]), 0)


def _compute_delta_f(delta_f, log_R, batch_size, lr, delta_f_max):
    if len(delta_f) == 1:
        delta_f = delta_f[0] - math.log(batch_size) - log_R
    else:
        delta_f = torch.logsumexp(torch.stack(delta_f), 0) - math.log(batch_size) - log_R

    delta_f[torch.where(delta_f.isnan())] = -float("Inf")
    delta_f = lr * torch.exp(delta_f)
    delta_f = torch.clamp(delta_f, -delta_f_max, delta_f_max)

    return delta_f



def SATRAM(dataset, f, log_v, lr, batch_size, delta_f_max, *args, **kwargs):

    if batch_size > dataset.dataloader.batch_size:
        batches_per_update = batch_size / dataset.dataloader.batch_size
    else:
        batches_per_update = 1

    log_v, log_R = compute_v_R(f, log_v, dataset.log_C_sym, dataset.state_counts, dataset.log_N)

    batch_updates_f = []

    for batch_idx, batch_data in enumerate(dataset.dataloader):
        batch_data = batch_data.to(dataset.device)

        batch_updates_f.append(_compute_batch_delta_f(f, log_R, batch_data[:, :dataset.n_therm_states],
                                                      batch_data[:, dataset.n_therm_states:]))

        if (batch_idx + 1) % batches_per_update == 0:

            delta_f = _compute_delta_f(batch_updates_f, log_R, batch_size, lr, delta_f_max)

            f_new = f - delta_f
            f = f_new - torch.min(f_new)
            batch_updates_f = []

            log_v, log_R = compute_v_R(f, log_v, dataset.log_C_sym, dataset.state_counts, dataset.log_N)

    return f, log_v



