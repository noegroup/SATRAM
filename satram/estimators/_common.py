import torch


def compute_sample_weights_batch(f, log_R, bias, ind_trajs):
    # TODO this is a workaround because we allowed empty states.
    f_plus_R = log_R + f
    f_plus_R[torch.where(log_R.isinf())] = torch.inf
    return -torch.logsumexp(f_plus_R - bias[:, :, None], 1) + torch.log(ind_trajs)


def compute_sample_weights(f, log_R, dataloader, device):
    log_weights = []

    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        log_weights.append(compute_sample_weights_batch(f, log_R, batch_data[:, :f.shape[0]],
                                                        batch_data[:, f.shape[0]:]))

    return torch.cat(log_weights)


def compute_v_R(f, log_v, log_C_sym, state_counts, log_N):
    log_Z_v_1 = log_v[:, None, :] - f[:, :, None]
    log_Z_v_2 = log_v[:, :, None] - f[:, None, :]
    log_Z_v_m = torch.maximum(log_Z_v_1, log_Z_v_2)
    log_Z_v = torch.log(torch.exp(log_Z_v_1 - log_Z_v_m) + torch.exp(log_Z_v_2 - log_Z_v_m)) + log_Z_v_m

    # torch doesn't like adding infinities.
    # set infinities to zero to get rid of NaNs in output. We can do this because adding log_C_sym will
    # revert these values back to negative inf.
    log_Z_v[torch.where(log_C_sym.isinf())] = 0

    log_v_new = torch.logsumexp(log_C_sym - f[:, None, :] + log_v[:, :, None]
                                - log_Z_v, 2) - log_N

    log_R = torch.logsumexp(log_C_sym - f[:, :, None] + log_v[:, None, :]
                            - log_Z_v, 2) - log_N

    log_R[torch.where(state_counts == 0)] = -float("Inf")
    log_v_new[torch.where(state_counts == 0)] = -float("Inf")

    return log_v_new, log_R


def compute_f_therm(f):
    f_therm = -torch.logsumexp(-f, 1)
    return f_therm - f_therm.min()

