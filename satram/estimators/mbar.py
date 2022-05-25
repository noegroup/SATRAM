from ._common import *


def _compute_batch_update_f(f, N_k_log, bias):
    return torch.logsumexp(-bias.T - torch.logsumexp(N_k_log - bias + f, axis=1), 1)


def _update_f(f, N_k_log, dataloader, device):
    f_new = []
    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        f_new.append(_compute_batch_update_f(f, N_k_log, batch_data[:, :f.shape[0]]))

    f_new = torch.logsumexp(torch.stack(f_new), axis=0)

    return -f_new


def MBAR(dataset, f, log_v, *args, **kwargs):
    # allow for two-dimensional free energies in case we want to do MBAR iterations as an initialization for TRAM.
    if f.ndim == 2:
        f_therm = compute_f_therm(f)
    else:
        f_therm = f

    f_therm = _update_f(f_therm, dataset.N_k_log, dataset.dataloader, dataset.device)

    if f.ndim == 2:
        f *= 0
        f += f_therm[:, None]
    else:
        f = f_therm
    return f, log_v