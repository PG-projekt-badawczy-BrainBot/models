import numpy as np
from scipy.linalg import fractional_matrix_power, logm
from pyriemann.utils import mean_riemann

def calculate_covariance(X):
    N, CH, L = X.shape
    factor = 1 / (L-1)
    print(f'factor: {factor}')
    res = np.empty((N, CH, CH))
    for idx, M in enumerate(X):
        C = factor * M@M.T
        res[idx] = C
    return res

def approximate_riemannian_mean(X: np.ndarray, tolerance = 1e-8):
    """
    Calculates approximate riemannian mean substituting it for log-gaussian mean

    Params:
        X (np.ndarray) - numpy array containing SPD matrices
    Returns:
        C - approximated Riemannian mean
    """
    C = mean_riemann(X, tol=tolerance)
    return C

def embed_into_tangent(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Embeds the matrices from X into a Riemannian tangent space based on the approximated affine invariant mean C

    Params:
        X (np.ndarray) - numpy array containing SPD matrices
        C (np.ndarray) - numpy array representing approximated affine-invariant mean of X
    Returns:
        X_r (np.ndarray) - numpy array containing X embedded in the tangent space
    """
    C_squared = fractional_matrix_power(C, 0.5)
    C_squared_minus = fractional_matrix_power(C, -0.5)
    X_r = np.empty(X.shape)
    for idx, C_i in enumerate(X):
        print(f'Embedding {idx+1} out of {X_r.shape[0]} matrices')
        inner = C_squared_minus@C_i@C_squared_minus
        log_inner = logm(inner)
        X_r[idx] = C_squared@log_inner@C_squared
    return X_r

def vectorize_tangent_matrices(X: np.ndarray) -> np.ndarray:
    """
    Vectorizes SPD matrices embedded in Riemannian tangent space in order to be able to compute gaussian metrics

    Params:
        X (np.ndarray) - numpy array containing matrices embedded into Riemannian tangent space

    Returns:
        V (np.ndarray) - numpy array of shape (n_samples, (N+1)*N/2)
    """
    N, CH, _ = X.shape
    shape = (N, (CH+1)*CH//2)
    V = np.empty(shape)
    mask = ~np.eye(CH, dtype=np.bool)
    factor = np.sqrt(2)
    for idx, M in enumerate(X):
        M[mask] *= factor
        lower_half = np.tril(M)
        symm_flat = lower_half[lower_half != 0]
        V[idx] = symm_flat
    return V

