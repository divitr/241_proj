import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from utils import load_results


def plot_cert_vs_risk(results, output_file=None):
    cert_values = np.array([r['cert'] for r in results])
    excess_risk = np.array([r['excess_risk'] for r in results])
    alpha_values = np.array([r['alpha'] for r in results])

    mask = (
        np.isfinite(cert_values) &
        np.isfinite(excess_risk) &
        (cert_values > 1e-8) &
        (excess_risk > 1e-8)
    )
    cert_values = cert_values[mask]
    excess_risk = excess_risk[mask]
    alpha_values = alpha_values[mask]

    lo = min(cert_values.min(), excess_risk.min())
    hi = max(cert_values.max(), excess_risk.max())
    lo, hi = lo * 0.8, hi * 1.2

    unique_alphas = sorted(set(alpha_values))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
    alpha_to_color = {a: cmap[i] for i, a in enumerate(unique_alphas)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for alpha in unique_alphas:
        m = (alpha_values == alpha)
        ax.scatter(
            cert_values[m], excess_risk[m],
            c=[alpha_to_color[alpha]],
            s=35, alpha=0.35, edgecolors='none',
            label=f'α = {alpha:.1f}'
        )

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.8, label='y = x (perfect)')
    ax.plot([lo, hi], [5*lo, 5*hi], 'r--', lw=1.5, alpha=0.6, label='y = 5x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel('Certificate (from empirical covariance)', fontsize=14)
    ax.set_ylabel('Population Excess Risk', fontsize=14)
    ax.set_title('Certificate vs Actual Population Risk', fontsize=16)

    ax.legend(loc='upper left', fontsize=10, frameon=True)

    ax.grid(True, which='both', ls='--', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot certificate vs risk')
    parser.add_argument('--input', type=str,
                        default='expts/results/cert_calibration.jsonl',
                        help='Input results file')
    parser.add_argument('--output', type=str,
                        default='figs/out/cert_vs_risk_scatter.png',
                        help='Output figure path')

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    plot_cert_vs_risk(results, output_file=args.output)


if __name__ == '__main__':
    main()
