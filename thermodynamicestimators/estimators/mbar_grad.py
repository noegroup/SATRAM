import torch

class MBARGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, free_energies, sampled_potentials, normalized_N_i):

        biased_potentials = sampled_potentials - free_energies

        logsumexp = torch.logsumexp(-biased_potentials, axis=1)

        objective_function = torch.mean(logsumexp) - torch.sum(free_energies * normalized_N_i)

        ctx.save_for_backward(biased_potentials, normalized_N_i, logsumexp)

        return objective_function


    @staticmethod
    def backward(ctx, grad_output):
        biased_potentials, normalized_N_i, logsumexp, = ctx.saved_tensors
        grad = torch.mean(torch.exp(-biased_potentials.T - logsumexp), dim=1) - normalized_N_i
        return grad, None, None