from ._common import *


def _compute_batch_update_f(f, log_R, bias, ind_trajs):
    return torch.logsumexp(-bias[:, :, None] - torch.logsumexp(log_R + f - bias[:, :, None], 1, keepdim=True) +
                           torch.log(ind_trajs[:, None, :]), 0)

def _compute_f(f, log_R, dataloader, device):
    f_batch = []
    # still use batches because the GPU is to small for the whole dataset
    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)

        f_batch.append(_compute_batch_update_f(f, log_R, batch_data[:, :f.shape[0]],
                                               batch_data[:, f.shape[0]:]))

    f = -torch.logsumexp(torch.stack(f_batch), 0)
    return f - f.min()



def TRAM(dataset, f, log_v, *args, **kwargs):
    log_v, log_R = compute_v_R(f, log_v, dataset.log_C_sym, dataset.state_counts, dataset.log_N)
    f = _compute_f(f, log_R, dataset.dataloader, dataset.device)
    return f, log_v



