import numpy as np
from scipy.linalg import qr


def make_powerlaw_cov(d, alpha, c=1.0, seed=None):
    rng = np.random.default_rng(seed)

    eigenvalues = c * np.arange(1, d + 1, dtype=float) ** (-alpha)

    random_matrix = rng.standard_normal((d, d))
    U, _ = qr(random_matrix)

    Sigma = U @ np.diag(eigenvalues) @ U.T

    return Sigma, eigenvalues


def sample_gaussian_design(N, Sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    d = Sigma.shape[0]

    Z = rng.standard_normal((N, d))

    L = np.linalg.cholesky(Sigma)
    X = Z @ L.T

    return X


def sample_test_data(N_test, Sigma, w_star, sigma2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_test = sample_gaussian_design(N_test, Sigma, rng)
    epsilon_test = rng.normal(0, np.sqrt(sigma2), N_test)
    y_test = X_test @ w_star + epsilon_test

    return X_test, y_test
