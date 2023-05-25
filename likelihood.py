"""
This module compute the likelihood of a VARFIMA(0,d,q) process
"""
import torch

from math import pi, pow

def compute_likelihood(Y, Yhat, V):
    """
    """
    r = Y.size()[0]
    T = Y.size()[1]

    s = torch.zeros(T)
    p = torch.zeros(T)
    
    for j in range(0, T):
        p[j] = torch.det(V[:, :, j])
        s[j] = torch.matmul(torch.matmul((Y[:, j] - Yhat[:, j]).reshape(1, -1), \
            torch.inverse(V[:, :, j])), (Y[:, j] - Yhat[:, j]).reshape(-1, 1))

    result = pow(2 * pi, - r * T / 2) * torch.div(torch.exp(- 0.5 * torch.sum(s)), \
        torch.sqrt(torch.prod(p)))

    return result
