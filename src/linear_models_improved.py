import numpy as np


def geometry_certificate_improved(X, lam, B, sigma2, use_true_sigma=None):
    N, d = X.shape

    Sigma_hat = (X.T @ X) / N

    Sigma = use_true_sigma if use_true_sigma is not None else Sigma_hat

    eigs = np.linalg.eigvalsh(Sigma)
    eigs = eigs[eigs > 1e-10]

    d_eff = np.sum(eigs / (eigs + lam))

    bias_resolvent = np.sum(eigs / (eigs + lam)**2)
    bias_term = lam**2 * B**2 * bias_resolvent

    bias_term_simple = lam * B**2 * d_eff

    variance_resolvent = np.sum(eigs**2 / (eigs + lam)**2)
    variance_term = (sigma2 / N) * variance_resolvent

    variance_term_simple = (sigma2 / N) * d_eff

    cert_full = bias_term + variance_term

    cert_simple = bias_term_simple + variance_term_simple

    breakdown = {
        'cert_full': cert_full,
        'cert_simple': cert_simple,
        'bias_full': bias_term,
        'bias_simple': bias_term_simple,
        'variance_full': variance_term,
        'variance_simple': variance_term_simple,
        'd_eff': d_eff,
        'bias_resolvent': bias_resolvent,
        'variance_resolvent': variance_resolvent,
    }

    return cert_full, breakdown


def geometry_certificate_calibrated(deff_hat, lam, B, sigma2, N, alpha=5.0):
    base = lam * B**2 + (sigma2 / N) * deff_hat

    cert = alpha * base

    return cert
