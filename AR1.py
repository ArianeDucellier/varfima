"""
This module finds the model parameters of an AR(1) time series
"""
import torch

def analytic(X):
    """
    Compute phi using the analytical formula
    that minimizes least squares.
    """
    phi = torch.sum(X[0:(N - 1)] * X[1:N]) / \
        torch.sum(torch.square(X[0:(N - 1)]))
    return phi

def step_least_squares(X, phi, alpha):
    """
    Compute one step of the gradient descent
    for the least square method
    """
    error = X.clone().detach()
    LS = torch.sum(torch.square(error - \
        phi * torch.cat((torch.zeros(1), X[0:(N - 1)]), 0)))
    LS.backward()
    dphi = phi.grad
    phi = phi - alpha * dphi
    phi.retain_grad()
    return (phi, LS)

def least_squares(X, max_iter, alpha):
    """
    Compute phi by minimizing least squares
    using the gradient
    """
    phi = torch.rand(1, requires_grad=True)
    i_iter = 0
    while (i_iter < max_iter):
        (phi, LS) = step_least_squares(X, phi, alpha)
        print('iteration {} - phi = {} - loss = {}'.format( \
            i_iter + 1, phi.detach().numpy()[0], LS.detach().numpy()))
        i_iter = i_iter + 1
    return (phi, LS)

if __name__ == '__main__':

    # Choose parameters
    sigma = 1.0 # Standard deviation
    phi = 0.8 # AR(1) parameter
    N = 1000 # Length of the time series
    max_iter = 20 # Number of iterations for the gradient descent
    alpha = 0.0001 # Step for the gradient descent

    # Set seed
    torch.manual_seed(1)

    # Generate time series
    Z = torch.normal(torch.zeros(N), sigma * torch.ones(N))
    X = torch.zeros(N)
    for i in range(1, N):
        X[i] = phi * X[i - 1] + Z[i]

    # Analytical
    phi = analytic(X)
    print('The value found with the analytical method is: ', phi.item())

    # Least squares + gradient descent
    phi, LS = least_squares(X, max_iter, alpha)
    print('The value found with the least squares + gradient method is: ', phi.detach().numpy()[0])
