import torch


def compute_v_R(f, log_v, log_C_sym, state_counts, log_N):
    log_Z_v_1 = log_v[:, None, :] - f[:, :, None]
    log_Z_v_2 = log_v[:, :, None] - f[:, None, :]
    log_Z_v_m = torch.maximum(log_Z_v_1, log_Z_v_2)
    log_Z_v = torch.log(torch.exp(log_Z_v_1 - log_Z_v_m) + torch.exp(log_Z_v_2 - log_Z_v_m)) + log_Z_v_m

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

