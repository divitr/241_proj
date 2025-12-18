import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data import make_powerlaw_cov, sample_gaussian_design, sample_test_data
from linear_models import (
    sample_w_star, ridge_fit, compute_risk,
    compute_train_mse, compute_test_mse, geometry_certificate
)
from spectra import effective_dimension, deff_from_empirical_cov
from utils import set_seed, save_result


def run_single_experiment(alpha, N, d, lambda_values, sigma2, B, c, seed, output_file):
    """Run a single certificate calibration experiment."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    Sigma, eigenvalues = make_powerlaw_cov(d, alpha, c=c, seed=seed)

    X_train = sample_gaussian_design(N, Sigma, rng)

    w_star = sample_w_star(Sigma, B, rng)

    epsilon = rng.normal(0, np.sqrt(sigma2), N)
    y_train = X_train @ w_star + epsilon

    N_test = 10000
    X_test, y_test = sample_test_data(N_test, Sigma, w_star, sigma2, rng)

    for lam in lambda_values:
        w_hat = ridge_fit(X_train, y_train, lam)

        risks = compute_risk(w_hat, w_star, Sigma, sigma2)
        test_mse = compute_test_mse(X_test, y_test, w_hat)
        train_mse = compute_train_mse(X_train, y_train, w_hat)

        deff_true = effective_dimension(eigenvalues, lam)
        deff_emp, eigs_emp = deff_from_empirical_cov(X_train, lam)

        cert = geometry_certificate(deff_emp, lam, B, sigma2, N, C1=1.0, C3=1.0)

        cert_C1_2 = geometry_certificate(deff_emp, lam, B, sigma2, N, C1=2.0, C3=1.0)
        cert_C3_2 = geometry_certificate(deff_emp, lam, B, sigma2, N, C1=1.0, C3=2.0)
        cert_C_both_2 = geometry_certificate(deff_emp, lam, B, sigma2, N, C1=2.0, C3=2.0)

        baseline_deff_over_N = deff_emp / N
        baseline_N_inv = 1.0 / N
        baseline_trace = np.sum(eigenvalues)

        result = {
            'expt': 'cert_calibration',
            'alpha': alpha,
            'N': N,
            'd': d,
            'lambda': lam,
            'sigma2': sigma2,
            'B': B,
            'c': c,
            'seed': seed,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'excess_risk': risks['excess_risk'],
            'total_risk': risks['total_risk'],
            'cert': cert,
            'cert_C1_2': cert_C1_2,
            'cert_C3_2': cert_C3_2,
            'cert_C_both_2': cert_C_both_2,
            'deff_true': deff_true,
            'deff_emp': deff_emp,
            'baseline_deff_over_N': baseline_deff_over_N,
            'baseline_N_inv': baseline_N_inv,
            'baseline_trace': baseline_trace
        }

        save_result(result, output_file)


def main():
    parser = argparse.ArgumentParser(description='Run certificate calibration experiments')
    parser.add_argument('--alpha_values', type=float, nargs='+',
                        default=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
                        help='Power-law exponents to sweep')
    parser.add_argument('--N_values', type=int, nargs='+',
                        default=[50, 100, 200, 400, 800],
                        help='Sample sizes to sweep')
    parser.add_argument('--d', type=int, default=200,
                        help='Dimension')
    parser.add_argument('--sigma2_values', type=float, nargs='+',
                        default=[0.01, 0.1],
                        help='Noise variance values to sweep')
    parser.add_argument('--B', type=float, default=1.0,
                        help='Sigma-norm bound on w_star')
    parser.add_argument('--lambda_values', type=float, nargs='+',
                        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                        help='Regularization parameters to sweep')
    parser.add_argument('--c', type=float, default=1.0,
                        help='Eigenvalue scaling constant')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds per configuration')
    parser.add_argument('--output', type=str,
                        default='expts/results/cert_calibration.jsonl',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')

    args = parser.parse_args()

    n_total = (len(args.alpha_values) * len(args.N_values) *
               len(args.sigma2_values) * args.n_seeds)

    print(f"Running {n_total} experiments (with {len(args.lambda_values)} lambda values each)...")
    print(f"Alpha values: {args.alpha_values}")
    print(f"N values: {args.N_values}")
    print(f"Sigma2 values: {args.sigma2_values}")
    print(f"Lambda values: {args.lambda_values}")
    print(f"d = {args.d}, B = {args.B}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {args.output}\n")

    count = 0
    for alpha in args.alpha_values:
        for N in args.N_values:
            for sigma2 in args.sigma2_values:
                for seed in range(args.n_seeds):
                    count += 1

                    if args.verbose:
                        print(f"[{count}/{n_total}] alpha={alpha:.2f}, N={N}, "
                              f"sigma2={sigma2:.3f}, seed={seed}")

                    run_single_experiment(
                        alpha=alpha,
                        N=N,
                        d=args.d,
                        lambda_values=args.lambda_values,
                        sigma2=sigma2,
                        B=args.B,
                        c=args.c,
                        seed=seed,
                        output_file=args.output
                    )

    print(f"Completed! Results saved to {args.output}")


if __name__ == '__main__':
    main()
