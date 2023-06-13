"""
This module finds the model parameters of an ARI(p) time series
"""
import numpy as np
import torch

def step_least_squares(X, phi, d, alpha):
    """
    Compute one step of the gradient descent
    for the least square method
    """
    p = phi.size()[0]
    error = X.clone().detach()
    for i in range(0, p):
        error = error - phi[i] * (torch.cat((torch.zeros(p), X[(p - 1 - i):(N - 1 - i)]), 0) - \
            torch.cat((torch.zeros(p), X[(p - i):(N - i)]), 0)
    for j in range(0, d):
        error = error + 
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
