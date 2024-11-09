import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def multivariateGaussian(X, mu, sigma2):
    """
    Compute the probability density function of a multivariate Gaussian distribution.
    
    Arguments:
    X -- Data points (n_samples, n_features)
    mu -- Mean vector (n_features,)
    sigma2 -- Covariance matrix (n_features, n_features)
    
    Returns:
    p -- Probability densities for each data point (n_samples,)
    """
    # Create a multivariate normal distribution
    dist = multivariate_normal(mean=mu, cov=sigma2)
    # Get probabilities for each point in X (multivariate normal PDF)
    return dist.pdf(X)

def visualizeFit(X, mu, sigma2):
    """
    Visualize the dataset and its estimated Gaussian distribution.
    
    Arguments:
    X -- The dataset (n_samples, n_features)
    mu -- The mean vector of the multivariate Gaussian (n_features,)
    sigma2 -- The covariance matrix (n_features, n_features)
    """
    
    # Create a meshgrid for the contour plot
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))

    # Prepare grid in the shape (n_points, 2) for evaluating PDF
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    
    # Compute Z values (the density of the Gaussian distribution) for the grid points
    Z = multivariateGaussian(grid_points, mu, sigma2)
    Z = Z.reshape(X1.shape)

    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1],  marker='.', alpha=0.7) 

    
    # Only display contours if Z doesn't contain infinities
    if np.sum(np.isinf(Z)) == 0:
        levels = np.power(10.0, np.arange(-20, 4, 3))
        plt.contour(X1, X2, Z, levels=levels, alpha=0.5, colors='purple')  # Plot the contour
        
    return plt
    
def multivariate_gaussian(X, mu, Sigma2):
    """
    Compute the probability density function of the multivariate Gaussian distribution.

    Parameters
    ----------
    X : ndarray
        Data points (n_samples, n_features)
    mu : ndarray
        Mean vector (n_features,)
    Sigma2 : ndarray
        Covariance matrix or variance vector (n_features, n_features) or (n_features,)

    Returns
    -------
    p : ndarray
        Probability density values for each data point (n_samples,)
    """
    k = len(mu)
    
    # If Sigma2 is a vector, convert it to a diagonal matrix
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)

    # Subtract mean vector from X
    X = X - mu.T
    
    # Compute p using the multivariate Gaussian formula
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(Sigma2) ** (-0.5) * \
        np.exp(-0.5 * np.sum(X @ np.linalg.pinv(Sigma2) * X, axis=1))

    return p