import json
import numpy as np
from pathlib import Path


def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_result(result_dict, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = convert_numpy_types(result_dict)

    with open(output_path, 'a') as f:
        f.write(json.dumps(result_dict) + '\n')


def load_results(input_file):
    results = []
    input_path = Path(input_file)

    if not input_path.exists():
        return results

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    return results


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def make_config(
    alpha_values,
    N_values,
    d=200,
    sigma2=0.1,
    B=1.0,
    lam=1e-6,
    n_seeds=5,
    c=1.0
):
    config = {
        'alpha_values': alpha_values,
        'N_values': N_values,
        'd': d,
        'sigma2': sigma2,
        'B': B,
        'lam': lam,
        'n_seeds': n_seeds,
        'c': c
    }
    return config


def aggregate_results(results, group_by, agg_fn='median'):
    from collections import defaultdict

    groups = defaultdict(list)

    for result in results:
        key = tuple(result[k] for k in group_by)
        groups[key].append(result)

    aggregated = {}

    for key, group in groups.items():
        group_dict = dict(zip(group_by, key))

        numeric_keys = set()
        for result in group:
            for k, v in result.items():
                if k not in group_by and isinstance(v, (int, float)):
                    numeric_keys.add(k)

        for num_key in numeric_keys:
            values = [r[num_key] for r in group if num_key in r]

            if agg_fn == 'median':
                group_dict[f'{num_key}_{agg_fn}'] = np.median(values)
            elif agg_fn == 'mean':
                group_dict[f'{num_key}_{agg_fn}'] = np.mean(values)
            elif agg_fn == 'std':
                group_dict[f'{num_key}_{agg_fn}'] = np.std(values)

        aggregated[key] = group_dict

    return aggregated
