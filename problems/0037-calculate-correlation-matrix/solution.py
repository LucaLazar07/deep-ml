import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Your code here
    if Y is None:
        Y = X
    
    Xc = X - np.mean(X, axis=0)
    Yc = Y - np.mean(Y, axis=0)

    covariance_matrix = (Xc.T @ Yc) / (X.shape[0] - 1)

    x_std = np.std(X, axis=0, ddof=1)

    y_std = np.std(Y, axis=0, ddof=1)
    
    return covariance_matrix / np.outer(x_std, y_std)