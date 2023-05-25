"""
This module compute the autocovariance of a VARFIMA(0, d, q) model
"""
import torch

import ACV_0d0

def compute_gamma_function(d, h, q):
    """
    Compute an auxiliary function containing the ratio
    of gamma functions of d, h and l with -q <= l <= q
    Input:
      d = 1D array, orders of the fractionally integrated process
      h = integer, lag time at which we compute the autocovariance matrix
      q = integer, order of the MA part of the fractionally integrated process
    Output:
      gamma_f = 3D array (r * r * 2q+1) where r is the length of d
    """
    assert isinstance(d, torch.Tensor), \
        'The orders of the fractionally integrated process should be a torch tensor'
    assert len(d.size()) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert isinstance(h, int), \
        'The lag of the autocovariance should be an integer'
    assert h >= 0, \
        'The lag should be higher or equal to 0'
    assert isinstance(q, int), \
        'The order of the MA part of the fractionally integrated process should be an integer'
    assert q >= 1, \
        'The order of the MA part of the fractionally integrated process should be higher or equal to 1'

    r = d.size()[0]
    gamma_f = torch.zeros(r, r, 2 * q + 1)
    gamma_f[:, :, q] = torch.ones((r, r)) # gamma_f(0)
    for m in range(0, r):
        for n in range(0, r):
            for k in range(1, q + 1):
                # gamma_f(-k)
                gamma_f[m, n, q - k] = gamma_f[m, n, q - k + 1] * (h - d[m] - k + 1) / (h + d[n] - k)
                # gamma_f(k)
                gamma_f[m, n, q + k] = gamma_f[m, n, q + k] * (h + d[n] + k - 1) / (h - d[m] + k)
    return gamma_f

def compute_autocovariance(sigma, theta, d, h):
    """
    Compute the autocovariance matrices up to lag h
    Input:
      sigma = 2D array, covariance of the white noise process
      d = 1D array, orders of the fractionally integrated process
      h = integer, lag time up to which we compute the autocovariance matrix
    Output:
      omega = 3D array, autocovariance matrix at lag 0, 1, ... , h
    """
    assert isinstance(sigma, torch.Tensor), \
        'The covariance of the white noise process should be a torch tensor'
    assert len(sigma.size()) == 2, \
        'The dimension of the covariance of the white noise process should be 2'
    assert sigma.size()[0] == sigma.size()[1], \
        'The covariance of the white noise process should be a square matrix'
    assert isinstance(theta, torch.Tensor), \
        'The coefficients of the MA process should be a torch tensor'
    assert len(theta.size()) == 3, \
        'The dimension of the coefficients of the MA process should be 3'
    assert theta.size()[0] == sigma.size()[1], \
        'The coefficients of the MA process for orders 1, 2, ... should be a square matrix'
    assert isinstance(d, torch.Tensor), \
        'The orders of the fractionally integrated process should be a torch tensor'
    assert len(d.size()) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert sigma.size()[0] == d.size()[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the covariance of the white noise process'
    assert theta.size()[0] == d.size()[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the coefficients of the MA process'
    assert isinstance(h, int), \
        'The maximum lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The maximum lag should be higher or equal to 1'

    r = d.size()[0]
    q = theta.size()[2] - 1
    omega = torch.zeros(r, r, h + 1)
    
    omega_star = ACV_0d0.compute_autocovariance(torch.eye(r), d, h)

    f0 = torch.matmul(torch.matmul(torch.diag(torch.diagonal(theta[:, :, 0])), sigma), \
                                   torch.diag(torch.diagonal(theta[:, :, 0])))

    for k in range(0, h + 1):
        gamma_f = compute_gamma_function(d, k, q)
    
        f1 = torch.zeros(r, r)
        f2 = torch.zeros(r, r)
        f3 = torch.zeros(r, r)
        for f in range(1, q + 1):
            for g in range(1, q + 1):
                f1 = f1 + torch.matmul(torch.matmul(theta[:, :, f], sigma), \
                    torch.transpose(theta[:, :, g], 0, 1)) * gamma_f[:, :, q + g - f]
            f2 = f2 + torch.matmul(theta[:, :, f], torch.transpose(sigma, 0, 1)) * \
                gamma_f[:, :, q - f]
            f3 = f3 + torch.matmul(sigma, torch.transpose(theta[:, :, f], 0, 1)) * \
                gamma_f[:, :, q + f]

        omega[:, :, k] = omega_star[:, :, k] * (f0 + f1 + f2 + f3)

    return omega
