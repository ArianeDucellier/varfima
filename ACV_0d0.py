"""
This module compute the autocovariance of a VARFIMA(0, d, 0) model
"""
import torch

def compute_next(omega_old, d, h):
    """
    Compute the autocovariance matrix at lag h
    from the autocovariance matrix at lag h-1
    Input:
      omega_old = 2D array, autocovariance matrix at lag h - 1
      d = 1D array, orders of the fractionally integrated process
      h = integer, lag time at which we compute the autocovariance matrix
    Output:
      omega = 2D array, autocovariance matrix at lag h
    """
    assert isinstance(omega_old, torch.Tensor), \
        'The autocovariance at lag h - 1 should be a torch tensor'
    assert len(omega_old.size()) == 2, \
        'The dimension of the autocovariance at lag h - 1 should be 2'
    assert omega_old.size()[0] == omega_old.size()[1], \
        'The autocovariance at lag h - 1 should be a square matrix'
    assert isinstance(d, torch.Tensor), \
        'The orders of the fractionally integrated process should be a torch Tensor'
    assert len(d.size()) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert omega_old.size()[0] == d.size()[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the autocovariance matrix at lag h - 1'
    assert isinstance(h, int), \
        'The lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The lag should be higher or equal to 1'

    r = d.size()[0]
    omega = torch.FloatTensor(r, r)
    for m in range(0, r):
        for n in range(0, r):
            omega[m, n] = omega_old[m, n] * (h - 1 + d[n]) / (h - d[m])
    return omega

def compute_autocovariance(sigma, d, h):
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
    assert isinstance(d, torch.Tensor), \
        'The orders of the fractionally integrated process should be a torch tensor'
    assert len(d.size()) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert sigma.size()[0] == d.size()[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the covariance of the white noise process'
    assert isinstance(h, int), \
        'The maximum lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The maximum lag should be higher or equal to 1'

    r = d.size()[0]
    omega = torch.FloatTensor(r, r, h + 1)
    for m in range(0, r):
        for n in range(0, r):
            omega[m, n, 0] = sigma[m, n] * torch.exp(torch.lgamma(1 - d[m] - d[n])) \
                / (torch.exp(torch.lgamma(1 - d[m])) * torch.exp(torch.lgamma(1 - d[n])))
    for k in range(1, h + 1):
        omega[:, :, k] = compute_next(omega[:, :, k - 1], d, h)
    return omega
