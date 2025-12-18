import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from utils import load_results


def plot_nn_cert_vs_risk(results, output_file=None):
    cert_ntk = np.array([r['cert_ntk'] for r in results])
    test_mse = np.array([r['test_mse'] for r in results])
    alpha_values = np.array([r['alpha'] for r in results])
    width_values = np.array([r['width'] for r in results])

    mask = (
        np.isfinite(cert_ntk) &
        np.isfinite(test_mse) &
        (cert_ntk > 1e-8) &
        (test_mse > 1e-8)
    )
    cert_ntk = cert_ntk[mask]
    test_mse = test_mse[mask]
    alpha_values = alpha_values[mask]
    width_values = width_values[mask]

    print(f"Plotting {len(cert_ntk)} valid points")

    lo = min(cert_ntk.min(), test_mse.min())
    hi = max(cert_ntk.max(), test_mse.max())
    lo, hi = lo * 0.8, hi * 1.2

    unique_alphas = sorted(set(alpha_values))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
    alpha_to_color = {a: cmap[i] for i, a in enumerate(unique_alphas)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for alpha in unique_alphas:
        m = (alpha_values == alpha)
        ax.scatter(
            cert_ntk[m], test_mse[m],
            c=[alpha_to_color[alpha]],
            s=35, alpha=0.35, edgecolors='none',
            label=f'α = {alpha:.1f}'
        )

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.8, label='y = x (perfect)')
    ax.plot([lo, hi], [3*lo, 3*hi], 'orange', linestyle='--', lw=1.5, alpha=0.6, label='y = 3x')
    ax.plot([lo, hi], [5*lo, 5*hi], 'r--', lw=1.5, alpha=0.6, label='y = 5x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    unique_widths = sorted(set(width_values))
    width_str = f"width={unique_widths[0]}" if len(unique_widths) == 1 else f"widths={unique_widths}"

    ax.set_xlabel('NTK Certificate (from initialization)', fontsize=14)
    ax.set_ylabel('Test MSE (held-out data)', fontsize=14)
    ax.set_title(f'Wide Neural Network: Certificate vs Test Risk\n({width_str})', fontsize=16)

    ax.legend(loc='upper left', fontsize=10, frameon=True)

    ax.grid(True, which='both', ls='--', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    else:
        plt.show()

    plt.close()

    ratio = test_mse / cert_ntk
    print(f"\nStatistics:")
    print(f"  Median ratio (Test MSE / Certificate): {np.median(ratio):.2f}×")
    print(f"  Mean ratio: {np.mean(ratio):.2f}×")
    print(f"  Std ratio: {np.std(ratio):.2f}×")

    from scipy.stats import pearsonr
    r_squared = pearsonr(np.log(cert_ntk), np.log(test_mse))[0]**2
    print(f"  R² (log-log): {r_squared:.3f}")


def plot_nn_vs_linear_cert(results, output_file=None):
    cert_ntk = np.array([r['cert_ntk'] for r in results])
    cert_linear = np.array([r['cert_linear'] for r in results])
    test_mse = np.array([r['test_mse'] for r in results])
    alpha_values = np.array([r['alpha'] for r in results])

    mask = (
        np.isfinite(cert_ntk) &
        np.isfinite(cert_linear) &
        np.isfinite(test_mse) &
        (cert_ntk > 1e-8) &
        (cert_linear > 1e-8) &
        (test_mse > 1e-8)
    )
    cert_ntk = cert_ntk[mask]
    cert_linear = cert_linear[mask]
    test_mse = test_mse[mask]
    alpha_values = alpha_values[mask]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    unique_alphas = sorted(set(alpha_values))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
    alpha_to_color = {a: cmap[i] for i, a in enumerate(unique_alphas)}

    ax = axes[0]
    for alpha in unique_alphas:
        m = (alpha_values == alpha)
        ax.scatter(
            cert_ntk[m], test_mse[m],
            c=[alpha_to_color[alpha]],
            s=35, alpha=0.35, edgecolors='none',
            label=f'α = {alpha:.1f}'
        )

    lo1 = min(cert_ntk.min(), test_mse.min()) * 0.8
    hi1 = max(cert_ntk.max(), test_mse.max()) * 1.2
    ax.plot([lo1, hi1], [lo1, hi1], 'k--', lw=1.5, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('NTK Certificate', fontsize=13)
    ax.set_ylabel('Test MSE', fontsize=13)
    ax.set_title('NTK Certificate vs Test Risk', fontsize=14)
    ax.legend(fontsize=9, loc='upper left', frameon=True)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    ax = axes[1]
    for alpha in unique_alphas:
        m = (alpha_values == alpha)
        ax.scatter(
            cert_linear[m], test_mse[m],
            c=[alpha_to_color[alpha]],
            s=35, alpha=0.35, edgecolors='none',
            label=f'α = {alpha:.1f}'
        )

    lo2 = min(cert_linear.min(), test_mse.min()) * 0.8
    hi2 = max(cert_linear.max(), test_mse.max()) * 1.2
    ax.plot([lo2, hi2], [lo2, hi2], 'k--', lw=1.5, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Linear Certificate (d_λ(Σ))', fontsize=13)
    ax.set_ylabel('Test MSE', fontsize=13)
    ax.set_title('Linear Certificate vs Test Risk', fontsize=14)
    ax.legend(fontsize=9, loc='upper left', frameon=True)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison figure to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot NN certificate vs risk')
    parser.add_argument('--input', type=str,
                        default='expts/results/nn_certificate.jsonl',
                        help='Input results file')
    parser.add_argument('--output', type=str,
                        default='figs/out/nn_cert_vs_risk.png',
                        help='Output figure path')
    parser.add_argument('--comparison_output', type=str,
                        default='figs/out/nn_vs_linear_cert.png',
                        help='Output for comparison figure')

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    plot_nn_cert_vs_risk(results, output_file=args.output)

    plot_nn_vs_linear_cert(results, output_file=args.comparison_output)


if __name__ == '__main__':
    main()
