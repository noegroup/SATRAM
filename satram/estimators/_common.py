import torch

epsl = 1e-10


def compute_f_therm(f):
    f_therm = -torch.logsumexp(-f, 1)
    return f_therm - f_therm.min()


def compute_sample_weights_batch(f, log_R, bias, ind_trajs):
    weights = -torch.logsumexp(f + log_R - bias[:, :, None], 1) + torch.log(ind_trajs)
    # TODO: find cleaner solution. NaNs get set to -inf because they are caused by adding infinities.
    # This is valid because the NaNs appear when log_R = -Inf and f=Inf. In that
    # case the weight for that state should be zero because there are no counts there.
    weights[torch.where(weights.isnan())] = -float("Inf")
    return torch.logsumexp(weights, 1)


def compute_sample_weights(f, log_R, dataloader, therm_state=None, device='cpu'):
    log_weights = []

    therm_state_energy = None
    if therm_state is not None:
        therm_state_energy = compute_f_therm(f)[therm_state]

    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        bias = batch_data[:, :f.shape[0]]
        ind_trajs = batch_data[:, f.shape[0]:]

        weights = compute_sample_weights_batch(f, log_R, bias, ind_trajs)

        if therm_state is not None:
            weights += therm_state_energy - bias[:, therm_state]

        log_weights.append(weights)

    return torch.cat(log_weights)


def compute_v_R(f, log_v, log_C_sym, log_N, state_counts, transition_counts):
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

    extra_counts = torch.log(epsl + state_counts
                             - transition_counts.transpose(1, 2).sum(2)) - log_N
    log_R = torch.logsumexp(torch.stack((log_R, extra_counts)), 0)
    log_R[torch.where(state_counts == 0)] = -torch.inf

    return log_v_new, log_R
