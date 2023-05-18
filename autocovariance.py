"""
This module compute the autocovariance of a VARFIMA(0, d, 0) model
"""
import numpy as np

from scipy.special import gamma

def compute_next(omega_old, d, h):
    """
    Compute the autocovariance matrix at lag h
    from the autocovariance matrix at lag h-1
    Input:
      omega_old = 2D numpy array, autocovariance matrix at lag h - 1
      d = 1D numpy array, orders of the fractionally integrated process
      h = scalar, lag time at which we compute the autocovariance matrix
    Output:
      omega = 2D numpy array, autocovariance matrix at lag h
    """
    assert isinstance(omega, np.ndarray), \
        'The autocovariance at lag h - 1 should be a numpy array'
    assert len(np.shape(omega_old)) == 2, \
        'The dimension of the autocovariance at lag h - 1 should be 2'
    assert np.shape(omega_old)[0] == np.shape(omega_old)[1], \
        'The autocovariance at lag h - 1 should be a square matrix'
    assert isinstance(d, np.ndarray), \
        'The orders of the fractionally integrated process should be a numpy array'
    assert len(np.shape(d)) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert np.shape(omega_old)[0] == np.shape(d)[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the autocovariance matrix at lag h - 1'
    assert isinstance(h, in), \
        'The lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The lag should be higher or equal to 1'

    r = np.shape(d)[0]
    omega = np.copy(omeda_old)
    for m in range(0, r):
        for n in range(0, r):
            omega[m, n] = omega_old[m, n] * (h - 1 + d[n]) / (h - d[m])
    return omega

def compute_autocovariance(sigma, d, h):
    """
    Compute the autocovariance matrices up to lag h
    Input:
      sigma = 2D numpy array, covariance of the white noise process
      d = 1D numpy array, orders of the fractionally integrated process
      h = scalar, lag time up to which we compute the autocovariance matrix
    Output:
      omega = 3D numpy array, autocovariance matrix at lag 0, 1, ... , h
    """
    assert isinstance(sigma, np.ndarray), \
        'The covariance of the white noise process should be a numpy array'
    assert len(np.shape(sigma)) == 2, \
        'The dimension of the covariance of the white noise process should be 2'
    assert np.shape(sigma)[0] == np.shape(sigma)[1], \
        'The covariance of the white noise process should be a square matrix'
    assert isinstance(d, np.ndarray), \
        'The orders of the fractionally integrated process should be a numpy array'
    assert len(np.shape(d)) == 1, \
        'The dimension of the orders of the fractionally integrated process should be 1'
    assert np.shape(sigma)[0] == np.shape(d)[0], \
        'The length of the orders of the fractionally integrated process should be ' + \
        'equal to the dimension of the covariance of the white noise process'
    assert isinstance(h, in), \
        'The maximum lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The maximum lag should be higher or equal to 1'

    r = np.shape(d)[0]
    omega = np.zeros((r, r, h + 1))
    for m in range(0, r):
        for n in range(0, r):
            omega[m, n, 0] = sigma[m, n] * gamma(1 - d[m] - d[n]) \
                / (gamma(1 - d[m]) * gamma(1 - d[n]))
    for k in range(1, h + 1):
        omega[:, :, k] = compute_next(omega[:, :, k - 1], d, h)
    return omega
