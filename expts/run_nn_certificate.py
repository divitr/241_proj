
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data import make_powerlaw_cov, sample_gaussian_design, sample_test_data
from linear_models import sample_w_star
from mlp_models import TwoLayerMLP, train_mlp, compute_mlp_test_mse
from ntk import compute_ntk_gram_batched, effective_dimension_ntk
from spectra import effective_dimension
from utils import set_seed, save_result


def ntk_certificate(K, lam, B, sigma2, N):
    deff_ntk, _ = effective_dimension_ntk(K, lam, N)

    bias_term = lam * B**2
    variance_term = (sigma2 / N) * deff_ntk
    cert = bias_term + variance_term

    return cert, deff_ntk


def run_single_experiment(
    alpha, N, d, width, sigma2, B, c, seed,
    lr, max_epochs, target_train_mse,
    lambda_values, output_file, device='cpu', verbose_train=False
):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    Sigma, eigenvalues = make_powerlaw_cov(d, alpha, c=c, seed=seed)

    X_train = sample_gaussian_design(N, Sigma, rng)

    w_star = sample_w_star(Sigma, B, rng)

    epsilon = rng.normal(0, np.sqrt(sigma2), N)
    y_train = X_train @ w_star + epsilon

    N_test = 2000
    X_test, y_test = sample_test_data(N_test, Sigma, w_star, sigma2, rng)

    model = TwoLayerMLP(d_in=d, width=width, d_out=1, activation='relu', init_scale=1.0)

    if N <= 512:
        print(f"  Computing NTK Gram matrix (N={N}, width={width})...")
        K_init = compute_ntk_gram_batched(model, X_train, device=device, batch_size=min(32, N))
    else:
        print(f"  Skipping NTK computation (N={N} too large)")
        K_init = None

    print(f"  Training MLP (width={width})...")
    train_losses, test_losses = train_mlp(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lr=lr,
        max_epochs=max_epochs,
        target_train_mse=target_train_mse,
        batch_size=None,
        device=device,
        verbose=verbose_train
    )

    final_train_mse = train_losses[-1]
    final_test_mse = test_losses[-1] if test_losses else compute_mlp_test_mse(model, X_test, y_test, device)

    for lam in lambda_values:
        if K_init is not None:
            cert_ntk, deff_ntk = ntk_certificate(K_init, lam, B, sigma2, N)
        else:
            cert_ntk, deff_ntk = np.nan, np.nan

        deff_linear = effective_dimension(eigenvalues, lam)
        cert_linear = lam * B**2 + (sigma2 / N) * deff_linear

        result = {
            'expt': 'nn_certificate',
            'alpha': alpha,
            'N': N,
            'd': d,
            'width': width,
            'sigma2': sigma2,
            'B': B,
            'c': c,
            'lambda': lam,
            'seed': seed,
            'lr': lr,
            'max_epochs': max_epochs,
            'n_epochs_trained': len(train_losses),
            'train_mse': final_train_mse,
            'test_mse': final_test_mse,
            'cert_ntk': cert_ntk,
            'cert_linear': cert_linear,
            'deff_ntk': deff_ntk,
            'deff_linear': deff_linear,
            'deff_ntk_over_N': deff_ntk / N if not np.isnan(deff_ntk) else np.nan,
            'deff_linear_over_N': deff_linear / N
        }

        save_result(result, output_file)


def main():
    parser = argparse.ArgumentParser(description='Run NN certificate experiments')
    parser.add_argument('--alpha_values', type=float, nargs='+',
                        default=[0.5, 0.8, 1.2, 1.5, 2.0],
                        help='Power-law exponents to sweep')
    parser.add_argument('--N_values', type=int, nargs='+',
                        default=[64, 128, 256],
                        help='Sample sizes to sweep')
    parser.add_argument('--d', type=int, default=50,
                        help='Input dimension')
    parser.add_argument('--width', type=int, default=4096,
                        help='Hidden layer width (wide network)')
    parser.add_argument('--sigma2', type=float, default=0.1,
                        help='Noise variance')
    parser.add_argument('--B', type=float, default=1.0,
                        help='Sigma-norm bound on w_star')
    parser.add_argument('--lambda_values', type=float, nargs='+',
                        default=[1e-4, 1e-3, 1e-2, 1e-1],
                        help='Regularization parameters for certificate')
    parser.add_argument('--c', type=float, default=1.0,
                        help='Eigenvalue scaling constant')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5000,
                        help='Maximum training epochs')
    parser.add_argument('--target_train_mse', type=float, default=None,
                        help='Target train MSE for early stopping')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of random seeds per configuration')
    parser.add_argument('--output', type=str,
                        default='expts/results/nn_certificate.jsonl',
                        help='Output file path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')
    parser.add_argument('--verbose_train', action='store_true',
                        help='Print training progress')

    args = parser.parse_args()

    n_total = len(args.alpha_values) * len(args.N_values) * args.n_seeds

    print(f"Running {n_total} NN Certificate experiments...")
    print(f"Alpha values: {args.alpha_values}")
    print(f"N values: {args.N_values}")
    print(f"d = {args.d}, width = {args.width} (WIDE NETWORK)")
    print(f"sigma2 = {args.sigma2}, B = {args.B}")
    print(f"Lambda values: {args.lambda_values}")
    print(f"Training: lr = {args.lr}, max_epochs = {args.max_epochs}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}\n")

    count = 0
    for alpha in args.alpha_values:
        for N in args.N_values:
            for seed in range(args.n_seeds):
                count += 1

                if args.verbose:
                    print(f"\n[{count}/{n_total}] alpha={alpha:.2f}, N={N}, seed={seed}")

                run_single_experiment(
                    alpha=alpha,
                    N=N,
                    d=args.d,
                    width=args.width,
                    sigma2=args.sigma2,
                    B=args.B,
                    c=args.c,
                    seed=seed,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    target_train_mse=args.target_train_mse,
                    lambda_values=args.lambda_values,
                    output_file=args.output,
                    device=args.device,
                    verbose_train=args.verbose_train
                )

    print(f"\nCompleted! Results saved to {args.output}")


if __name__ == '__main__':
    main()
