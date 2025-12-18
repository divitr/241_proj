import numpy as np


def sample_w_star(Sigma, B, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    d = Sigma.shape[0]

    w_raw = rng.standard_normal(d)

    sigma_norm_sq = w_raw @ Sigma @ w_raw

    w_star = B * w_raw / np.sqrt(sigma_norm_sq)

    return w_star


def ridge_fit(X, y, lam):
    N, d = X.shape

    gram = X.T @ X / N
    cross = X.T @ y / N

    w_hat = np.linalg.solve(gram + lam * np.eye(d), cross)

    return w_hat


def compute_risk(w_hat, w_star, Sigma, sigma2):
    delta = w_hat - w_star

    excess_risk = delta @ Sigma @ delta

    total_risk = sigma2 + excess_risk

    param_error = delta @ delta

    return {
        'excess_risk': excess_risk,
        'total_risk': total_risk,
        'param_error': param_error
    }


def compute_train_mse(X, y, w_hat):
    y_pred = X @ w_hat
    return np.mean((y - y_pred) ** 2)


def compute_test_mse(X_test, y_test, w_hat):
    y_pred = X_test @ w_hat
    return np.mean((y_test - y_pred) ** 2)


def geometry_certificate(deff_hat, lam, B, sigma2, N, C1=1.0, C3=1.0):
    bias_term = C1 * lam * B**2
    variance_term = C3 * (sigma2 / N) * deff_hat
    cert = bias_term + variance_term

    return cert
