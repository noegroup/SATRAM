import torch

class MBARGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, free_energies, sampled_potentials, normalized_N_i):

        sumexp = torch.sum(torch.exp(-sampled_potentials + free_energies), axis=1)

        logsum = torch.log(sumexp)

        objective_function = torch.mean(logsum) - torch.sum(free_energies * normalized_N_i)

        ctx.save_for_backward(free_energies, sampled_potentials, normalized_N_i, sumexp)

        return objective_function


    @staticmethod
    def backward(ctx, grad_output):
        free_energies, sampled_potentials, normalized_N_i, sumexp, = ctx.saved_tensors
        grad = torch.mean(torch.exp(-sampled_potentials + free_energies).T / sumexp, dim=1) - normalized_N_i
        return grad, None, None