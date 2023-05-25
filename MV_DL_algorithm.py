"""
This module uses the multivariate Durbin-Levinson algorithm
to compute the best linear predictor and the corresponding covariance.
See Proposition 11.4.1 of Brockwell, P. J. and Davis, R. A., 1991,
Time Series: Theory and Methods, 2nd Ed. Springer-Verlag, New York for the formulae.
"""
import torch

def init_DL(omega):
    """
    Initiate the innovation algorithm with
    Xhat_1 = 0 and V_0 = K(1,1) = Omega(0)
    Input:
      omega = 2D array, autocovariance matrix at lag 0
    Output:
      Xhat_1 = 1D array, best linear predictor at time step n = 1
      V_0 = 2D array, prediction error matrix at time step n = 0
    """
    assert isinstance(omega, torch.Tensor), \
        'The autocovariance should be a torch tensor'
    assert len(omega.size()) == 3, \
        'The dimension of the autocovariance at eaach lag should be 2'
    assert omega.size()[0] == omega.size()[1], \
        'The autocovariance should be a square matrix'

    r = omega.size()[0]
    Xhat_1 = torch.zeros(r)
    V_0 = omega[:, :, 0]
    return (Xhat_1, V_0)

def next_step(X_n, Xhat_n, Vnm1, omega, theta):
    """
    Compute the next best linear predictor and the corrresponding
    prediction error matrix at step n + 1
    Input:
      X_n = 2D array, observations at step 1, 2 , ... , n
      Xhat_n = 2D array, best linear predictor at steps 1, 2, ... , n
      Vnm1 = 3D array, prediction error at steps 0 , 1 , ... , n - 1
      omega = 3D array, autocovariance matrix at lags 0, 1, ... , n
      theta = 4D array, intermediate matrix at steps
          1-1, 2-1, 2-2, ... , n-1, n-n
    Output:
      Xhat_np1 = 1D array, best linear predictor at time step n + 1
      Vn = 2D array, prediction error at time step n
    """
    
    r = X_n.size()[0]
    n = X_n.size()[1]

    Xhat_np1 = torch.zeros(r)
    for j in range(1, n + 1):
        Xhat_np1 = Xhat_np1 + torch.matmul(theta[:, :, n - 1, j - 1], \
            X_n[:, n - j] - Xhat_n[:, n - j])

    for k in range(n - 1, - 1, -1):
        if k >= 1:
#        theta[:, :, n - 1, n - k - 1] = omega[:, :, n - k]
            theta_int = torch.zeros(r, r, k)
            for j in range(0, k):
                theta_int[:, :, j] = torch.matmul(torch.matmul(theta[:, :, n - 1, n - j - 1], \
                    Vnm1[:, :, j]), torch.transpose(theta[:, :, k, k - j], 0, 1))
                
#                theta[:, :, n - 1, n - k - 1] = theta[:, :, n - 1, n - k - 1] - \
#                    torch.matmul(torch.matmul(theta[:, :, n - 1, n - j - 1], \
#                    Vnm1[:, :, j]), torch.transpose(theta[:, :, k, k - j], 0, 1))

            theta[:, :, n - 1, n - k - 1] = torch.matmul(omega[:, :, n - k] - \
                torch.sum(theta_int, 2), torch.inverse(Vnm1[:, :, k]))
#        theta[:, :, n - 1, n - k - 1] = torch.matmul(theta[:, :, n - 1, n - k - 1], \
#            torch.inverse(Vnm1[:, :, k]))

        else:
            theta[:, :, n - 1, n - k - 1] = torch.matmul(omega[:, :, n - k], \
                torch.inverse(Vnm1[:, :, k]))
            
#    Vn = omega[:, :, 0]
    Vn_int = torch.zeros(r, r, n)
    for j in range(0, n):
        Vn_int[:, :, j] = torch.matmul(torch.matmul(theta[:, :, n - 1, n - j - 1], \
            Vnm1[:, :, j]), torch.transpose(theta[:, :, n - 1, n - j - 1], 0, 1))
    Vn = omega[:, :, 0] - torch.sum(Vn_int, 2)
#        Vn = Vn - torch.matmul(torch.matmul(theta[:, :, n - 1, n - j - 1], \
#            Vnm1[:, :, j]), torch.transpose(theta[:, :, n - 1, n - j - 1], 0, 1))

    return (Xhat_np1, Vn, theta)

def DL_alg(X, omega):
    """
    Compute the DL algorithm
    """
    r = X.size()[0]
    N = X.size()[1]

    Xhat = torch.zeros(r, N + 1)
    V = torch.zeros(r, r, N + 1)
    theta = torch.zeros(r, r, N, N)

    (Xhat_1, V_0) = init_DL(omega)
    Xhat[:, 0] = Xhat_1
    V[:, :, 0] = V_0
    for n in range(1, N + 1):
        X_n = X[:, 0:n]
        Xhat_n = Xhat[:, 0:n]
        Vnm1 = V[:, :, 0:n]  
        (Xhat_np1, Vn, theta) = next_step(X_n, Xhat_n, Vnm1, omega, theta)
        Xhat[:, n] = Xhat_np1
        V[:, :, n] = Vn

    return (Xhat, V)
