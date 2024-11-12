import numpy as np
import matplotlib.pyplot as plt

# Assume plot-data helper function
def plotData(X, y, labels=[0, 1]):
    """
    Plot data points with `+` for positive and `o` for negative examples.
    X is assumed to be Mx2 matrix.
    """
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='+', label=labels[1], color='b')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label=labels[0], color='r')

# This is used when the feature space is mapped into a higher-dimension, usually for nonlinear decision boundaries.
def mapFeature(u, v, degree = 6):
    """
    Returns mapped features as necessary for higher dimensional decision boundary.
    You'll need to implement based on how you're generating polynomial features or higher dimensions.
    Adapt this accordingly to match exactly with your `mapFeature` routine.
    """
    # Example: Polynomial mapping for u and v up to degree 6.
    out = np.ones(1)
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.append(out, (u ** (i - j)) * (v ** j))
    return out

def plotDecisionBoundary(theta, X, y, degree=6):
    """
    Plots the decision boundary for given theta values, using X and y.
    """

    if X.shape[1] <= 3:  
        plotData(X[:, 0:2], y, labels=["Not Admitted", "Admitted"])
        # Simple linear boundary case
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1.0 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot the decision boundary
        plt.plot(plot_x, plot_y, label='Decision Boundary')
        plt.legend()
        plt.axis([30, 100, 30, 100])

    else:
        plotData(X[:, 0:2], y)
        # Nonlinear boundary (e.g., if using higher-degree feature mapping)
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate the decision boundary for the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(mapFeature(u[i], v[j], degree), theta)

        z = z.T  # Transpose z, as in the original MATLAB code

        # Plot contour where z = 0 (the decision boundary)
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.legend()
    return plt