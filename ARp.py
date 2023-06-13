"""
This module finds the model parameters of an AR(p) time series
"""
import numpy as np
import torch

def analytic(X, p):
    """
    Compute phi using the analytical formula
    that minimizes least squares.
    """
    A = torch.zeros(p, p)
    B = torch.zeros(p)
    for i in range(0, p):
        B[i] = torch.sum(X[(i + 1):N] * X[0:(N - i - 1)])
        for j in range(0, p):
            if i > j:
                A[i, j] = torch.sum(X[0:(N - i - 1)] * X[(i - j):(N - j - 1)])
            else:
                A[i, j] = torch.sum(X[0:(N - j - 1)] * X[(j - i):(N - i - 1)])
    phi = torch.linalg.solve(A, B)
    return phi

def step_least_squares(X, phi, alpha):
    """
    Compute one step of the gradient descent
    for the least square method
    """
    p = phi.size()[0]
    error = X.clone().detach()
    for i in range(0, p):
        error = error - phi[i] * torch.cat((torch.zeros(p), X[(p - 1 - i):(N - 1 - i)]), 0)
    LS = torch.sum(torch.square(error))
    LS.backward()
    dphi = phi.grad
    phi = phi - alpha * dphi
    phi.retain_grad()
    return (phi, LS)

def least_squares(X, p, max_iter, alpha):
    """
    Compute phi by minimizing least squares
    using the gradient
    """
    phi = torch.rand(p, requires_grad=True)
    i_iter = 0
    while (i_iter < max_iter):
        (phi, LS) = step_least_squares(X, phi, alpha)
        print('iteration {} - phi = {} - loss = {}'.format( \
            i_iter + 1, phi.detach().numpy(), LS.detach().numpy()))
        i_iter = i_iter + 1
    return (phi, LS)

if __name__ == '__main__':

    # Choose parameters
    sigma = 1.0 # Standard deviation
    phi = np.array([0.8, 0.1]) # AR(p) parameter
    N = 1000 # Length of the time series
    max_iter = 100 # Number of iterations for the gradient descent
    alpha = 0.00001 # Step for the gradient descent

    # Set seed
    torch.manual_seed(1)

    # Generate time series
    Z = torch.normal(torch.zeros(N), sigma * torch.ones(N))
    X = Z.clone().detach()
    p = len(phi)
    for i in range(0, N):
        for j in range(0, p):
            if i > j:
                X[i] = X[i] + phi[j] * X[i - j - 1]

    # Analytical
    phi = analytic(X, p)
    print('The value found with the analytical method is: ', phi.detach().numpy())

    # Least squares + gradient descent
    phi, LS = least_squares(X, p, max_iter, alpha)
    print('The value found with the least squares + gradient method is: ', phi.detach().numpy())
