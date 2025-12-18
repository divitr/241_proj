import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils import load_results


def compute_roc_auc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(results, threshold=0.05, output_file=None):
    valid_results = [r for r in results
                     if np.isfinite(r['excess_risk'])
                     and np.isfinite(r['cert'])]

    y_true = np.array([1 if r['excess_risk'] <= threshold else 0 for r in valid_results])

    print(f"Benign samples: {np.sum(y_true)} / {len(y_true)} "
          f"({100*np.mean(y_true):.1f}%)")

    predictors = {
        'Certificate': -np.array([r['cert'] for r in valid_results]),
        'Certificate (C=2)': -np.array([r.get('cert_C_both_2', r['cert']) for r in valid_results]),
        'd_eff/N': -np.array([r['baseline_deff_over_N'] for r in valid_results]),
        '1/N': np.array([r['baseline_N_inv'] for r in valid_results]),
        'Trace($\\Sigma$)': -np.array([r['baseline_trace'] for r in valid_results]),
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, scores) in enumerate(predictors.items()):
        fpr, tpr, roc_auc = compute_roc_auc(y_true, scores)

        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2.5,
            alpha=0.85,
            label=f'{name} (AUC = {roc_auc:.3f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.500)', alpha=0.6)

    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'Predicting Benign Interpolation (threshold = {threshold})', fontsize=16)

    ax.legend(fontsize=10, loc='lower right', frameon=True)

    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot ROC curves for benign classification')
    parser.add_argument('--input', type=str,
                        default='expts/results/cert_calibration.jsonl',
                        help='Input results file')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Threshold for defining benign regime')
    parser.add_argument('--output', type=str,
                        default='figs/out/cert_roc_curve.png',
                        help='Output figure path')

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    plot_roc_curves(results, threshold=args.threshold, output_file=args.output)

if __name__ == '__main__':
    main()
