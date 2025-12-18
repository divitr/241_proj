
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))   

import numpy as np
import torch
from data import make_powerlaw_cov, sample_gaussian_design, sample_test_data
from linear_models import sample_w_star
from mlp_models import TwoLayerMLP, train_mlp, compute_mlp_test_mse
from ntk import compute_ntk_gram_batched, effective_dimension_ntk
from spectra import effective_dimension
from utils import set_seed, save_result


def run_single_experiment(
    alpha, N, d, width, sigma2, B, c, seed,
    lr, max_epochs, target_train_mse,
    lam, output_file, device='cpu', verbose_train=False
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

    print(f"  Computing NTK Gram matrix (N={N})...")
    K_init = compute_ntk_gram_batched(model, X_train, device=device, batch_size=min(32, N))

    deff_ntk_init, eigs_ntk_init = effective_dimension_ntk(K_init, lam, N)

    print(f"  Training MLP...")
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

    deff_linear = effective_dimension(eigenvalues, lam)

    result = {
        'expt': 'mlp_ntk',
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
        'target_train_mse': target_train_mse,
        'n_epochs_trained': len(train_losses),
        'train_mse': final_train_mse,
        'test_mse': final_test_mse,
        'deff_ntk_init': deff_ntk_init,
        'deff_linear': deff_linear,
        'deff_ntk_over_N': deff_ntk_init / N,
        'deff_linear_over_N': deff_linear / N
    }

    save_result(result, output_file)

    return result


def main():
    parser = argparse.ArgumentParser(description='Run MLP/NTK experiments')
    parser.add_argument('--alpha_values', type=float, nargs='+',
                        default=[0.5, 0.8, 1.2, 1.5, 2.0],
                        help='Power-law exponents to sweep')
    parser.add_argument('--N_values', type=int, nargs='+',
                        default=[64, 128, 256, 512],
                        help='Sample sizes to sweep')
    parser.add_argument('--d', type=int, default=50,
                        help='Input dimension')
    parser.add_argument('--width', type=int, default=1024,
                        help='Hidden layer width')
    parser.add_argument('--sigma2', type=float, default=0.1,
                        help='Noise variance')
    parser.add_argument('--B', type=float, default=1.0,
                        help='Sigma-norm bound on w_star')
    parser.add_argument('--lam', type=float, default=1e-3,
                        help='Regularization parameter for effective dimension')
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
                        default='expts/results/mlp_ntk.jsonl',
                        help='Output file path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')
    parser.add_argument('--verbose_train', action='store_true',
                        help='Print training progress')

    args = parser.parse_args()
    
    n_total = len(args.alpha_values) * len(args.N_values) * args.n_seeds

    print(f"Running {n_total} MLP/NTK experiments...")
    print(f"Alpha values: {args.alpha_values}")
    print(f"N values: {args.N_values}")
    print(f"d = {args.d}, width = {args.width}")
    print(f"sigma2 = {args.sigma2}, B = {args.B}, lambda = {args.lam}")
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

                result = run_single_experiment(
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
                    lam=args.lam,
                    output_file=args.output,
                    device=args.device,
                    verbose_train=args.verbose_train
                )

                if args.verbose:
                    print(f"  Train MSE: {result['train_mse']:.6f}, "
                          f"Test MSE: {result['test_mse']:.6f}")
                    print(f"  d_eff(NTK)/N: {result['deff_ntk_over_N']:.4f}, "
                          f"d_eff(linear)/N: {result['deff_linear_over_N']:.4f}")
                    print(f"  Epochs: {result['n_epochs_trained']}")

    print(f"\nCompleted! Results saved to {args.output}")


if __name__ == '__main__':
    main()
