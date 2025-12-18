import numpy as np


def effective_dimension(eigs, lam):
    deff = np.sum(eigs / (eigs + lam))
    return deff


def effective_rank(eigs):
    reff = np.sum(eigs) / np.max(eigs)
    return reff


def empirical_covariance(X):
    N = X.shape[0]
    Sigma_hat = (X.T @ X) / N
    return Sigma_hat


def deff_from_empirical_cov(X, lam):
    Sigma_hat = empirical_covariance(X)
    eigs_hat = np.linalg.eigvalsh(Sigma_hat)
    eigs_hat = eigs_hat[eigs_hat > 0]
    deff_hat = effective_dimension(eigs_hat, lam)

    return deff_hat, eigs_hat


def stable_rank(eigs):
    srank = np.sum(eigs**2) / (np.max(eigs)**2)
    return srank


def trace_inv(eigs, lam):
    trace = np.sum(1.0 / (eigs + lam))
    return trace


def participation_ratio(eigs):
    pr = (np.sum(eigs)**2) / np.sum(eigs**2)
    return pr
