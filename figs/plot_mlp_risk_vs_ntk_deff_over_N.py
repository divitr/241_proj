import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from utils import load_results


def plot_mlp_risk_vs_ntk_deff(results, output_file=None):
    deff_ntk_over_N = np.array([r['deff_ntk_over_N'] for r in results])
    test_mse = np.array([r['test_mse'] for r in results])
    alpha_values = np.array([r['alpha'] for r in results])

    mask = (
        np.isfinite(deff_ntk_over_N) &
        np.isfinite(test_mse) &
        (deff_ntk_over_N > 1e-8) &
        (test_mse > 1e-8)
    )
    deff_ntk_over_N = deff_ntk_over_N[mask]
    test_mse = test_mse[mask]
    alpha_values = alpha_values[mask]

    unique_alphas = sorted(set(alpha_values))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
    alpha_to_color = {alpha: colors[i] for i, alpha in enumerate(unique_alphas)}

    fig, ax = plt.subplots(figsize=(10, 7))

    for alpha in unique_alphas:
        m = (alpha_values == alpha)
        ax.scatter(
            deff_ntk_over_N[m], test_mse[m],
            c=[alpha_to_color[alpha]],
            label=f'α = {alpha:.1f}',
            alpha=0.35,
            s=35,
            edgecolors='none'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'NTK Effective Dimension / Sample Size: $d_\lambda(K_{NTK}) / N$', fontsize=14)
    ax.set_ylabel('Test MSE (held-out data)', fontsize=14)
    ax.set_title(r'MLP Risk Collapse: Test Risk vs NTK Effective Dimension', fontsize=16)

    ax.legend(fontsize=10, loc='upper left', frameon=True)

    ax.grid(True, which='both', ls='--', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot MLP risk vs NTK d_eff/N')
    parser.add_argument('--input', type=str,
                        default='expts/results/mlp_ntk.jsonl',
                        help='Input results file')
    parser.add_argument('--output', type=str,
                        default='figs/out/mlp_risk_vs_ntk_deff_over_N.png',
                        help='Output figure path')

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    plot_mlp_risk_vs_ntk_deff(results, output_file=args.output)


if __name__ == '__main__':
    main()
