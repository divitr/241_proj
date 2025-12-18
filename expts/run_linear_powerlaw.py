import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data import make_powerlaw_cov, sample_gaussian_design, sample_test_data
from linear_models import sample_w_star, ridge_fit, compute_risk, compute_train_mse, compute_test_mse
from spectra import effective_dimension, effective_rank
from utils import set_seed, save_result


def run_single_experiment(alpha, N, d, lam, sigma2, B, c, seed, output_file):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    Sigma, eigenvalues = make_powerlaw_cov(d, alpha, c=c, seed=seed)

    X_train = sample_gaussian_design(N, Sigma, rng)

    w_star = sample_w_star(Sigma, B, rng)

    epsilon = rng.normal(0, np.sqrt(sigma2), N)
    y_train = X_train @ w_star + epsilon

    w_hat = ridge_fit(X_train, y_train, lam)

    train_mse = compute_train_mse(X_train, y_train, w_hat)

    N_test = 10000
    X_test, y_test = sample_test_data(N_test, Sigma, w_star, sigma2, rng)

    test_mse = compute_test_mse(X_test, y_test, w_hat)

    risks = compute_risk(w_hat, w_star, Sigma, sigma2)

    deff = effective_dimension(eigenvalues, lam)
    reff = effective_rank(eigenvalues)

    result = {
        'expt': 'linear_powerlaw',
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
        'param_error': risks['param_error'],
        'deff': deff,
        'reff': reff,
        'deff_over_N': deff / N
    }

    save_result(result, output_file)

    return result


def main():
    parser = argparse.ArgumentParser(description='Run linear power-law experiments')
    parser.add_argument('--alpha_values', type=float, nargs='+',
                        default=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
                        help='Power-law exponents to sweep')
    parser.add_argument('--N_values', type=int, nargs='+',
                        default=[50, 100, 200, 400, 800, 1600],
                        help='Sample sizes to sweep')
    parser.add_argument('--d', type=int, default=200,
                        help='Dimension')
    parser.add_argument('--sigma2', type=float, default=0.1,
                        help='Noise variance')
    parser.add_argument('--B', type=float, default=1.0,
                        help='Sigma-norm bound on w_star')
    parser.add_argument('--lam', type=float, default=1e-6,
                        help='Regularization parameter (ridgeless regime)')
    parser.add_argument('--c', type=float, default=1.0,
                        help='Eigenvalue scaling constant')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds per configuration')
    parser.add_argument('--output', type=str,
                        default='expts/results/linear_powerlaw.jsonl',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')

    args = parser.parse_args()

    n_total = len(args.alpha_values) * len(args.N_values) * args.n_seeds

    print(f"Running {n_total} experiments...")
    print(f"Alpha values: {args.alpha_values}")
    print(f"N values: {args.N_values}")
    print(f"d = {args.d}, sigma2 = {args.sigma2}, B = {args.B}, lambda = {args.lam}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {args.output}\n")

    count = 0
    for alpha in args.alpha_values:
        for N in args.N_values:
            for seed in range(args.n_seeds):
                count += 1

                if args.verbose:
                    print(f"[{count}/{n_total}] alpha={alpha:.2f}, N={N}, seed={seed}")

                result = run_single_experiment(
                    alpha=alpha,
                    N=N,
                    d=args.d,
                    lam=args.lam,
                    sigma2=args.sigma2,
                    B=args.B,
                    c=args.c,
                    seed=seed,
                    output_file=args.output
                )

                if args.verbose:
                    print(f"  Train MSE: {result['train_mse']:.6f}, "
                          f"Test MSE: {result['test_mse']:.6f}, "
                          f"Excess Risk: {result['excess_risk']:.6f}")
                    print(f"  d_eff/N: {result['deff_over_N']:.4f}\n")

    print(f"Completed! Results saved to {args.output}")


if __name__ == '__main__':
    main()
