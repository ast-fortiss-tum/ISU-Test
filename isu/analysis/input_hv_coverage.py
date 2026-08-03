import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from pymoo.indicators.hv import HV 
import time
from pymoo.config import Config
Config.warnings['not_compiled'] = False
from pathlib import Path
import matplotlib.pyplot as plt
from isu.config import SEARCH_SPACE_JSON

def load_search_space(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def encode_instance_to_vector(params, search_space):
    vec = []

    # ---- categorical_features ----
    for feat in search_space.get("categorical_features", []):
        name = feat["name"]
        values = feat["values"]
        all_vals_str = [str(v) for v in values]

        if name not in params.keys():
            print(f"name: {name}, params: {params.keys()}")
            raise ValueError(f"Missing categorical feature '{name}' in instance.")

        v_str = str(params[name])
        if v_str not in all_vals_str:
            raise ValueError(
                f"Value '{v_str}' not in search_space categorical values of '{name}': {all_vals_str}"
            )

        idx = all_vals_str.index(v_str)
        if len(all_vals_str) == 1:
            norm = 0.0
        else:
            norm = idx / (len(all_vals_str) - 1)
        vec.append(norm)

    # # ---- continuous_features ----
    # for feat in search_space.get("continuous_features", []):
    #     name = feat["name"]
    #     lb = float(feat["lb"])
    #     ub = float(feat["ub"])

    #     if name not in params:
    #         raise ValueError(f"Missing continuous feature '{name}' in instance.")

    #     v = float(params[name])
    #     if ub == lb:
    #         norm = 0.0
    #     else:
    #         norm = (v - lb) / (ub - lb)

    #     norm = max(0.0, min(1.0, norm))
    #     vec.append(norm)

    #only consider selected discrete features
    dominant_feature_indexes = [0, 1, 3, 4, 5, 6, 8, 10, 13, 15]
    #dominant_feature_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]
    vec = [vec[i] for i in dominant_feature_indexes]

    return np.array(vec, dtype=float)


def load_all_instances(inst_dir, search_space):
    vectors = []
    file_count = 0

    for fname in os.listdir(inst_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(inst_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            params = json.load(f)

        try:
            vec = encode_instance_to_vector(params, search_space)
        except Exception as e:
            print(f"[Skip] {fname}: {e}")
            continue

        vectors.append(vec)
        file_count += 1

    print(f"Loaded {file_count} instances from {inst_dir}")

    X = np.vstack(vectors)
    return X


def compute_hv_coverage(points, eps=1e-1):
    if points.ndim != 2:
        raise ValueError("points must be 2D array")

    pts = eps + (1.0 - 2.0 * eps) * points

    Y = 1.0 - pts

    ref = np.ones(points.shape[1])

    hv_indicator = HV(ref_point=ref)
    value = hv_indicator(Y)
    return float(value)


def calculate_input_hv_coverage(result_dir, start = 50, step = 100):
    space = load_search_space(SEARCH_SPACE_JSON)

    X = load_all_instances(Path(result_dir) / "json_inputs", space)
    #print(X.shape)

    # n = 10
    # time_start = time.time()
    # hv = compute_hv_coverage(X[:n])
    # time_end = time.time()
    # print(f"Hypervolume coverage for first {n} points: {hv:.2e}")
    # print(f"Computed in {time_end - time_start:.2f} seconds.")

    #per_dim = hv ** (1.0 / X.shape[1])
    #print(f"Approx per-dimension coverage: {per_dim:.2f}")

    #draw a plot of sample/coverage curve
    #including last point
    sample_sizes = list(range(start, X.shape[0], step))
    if sample_sizes[-1] != X.shape[0]:
        sample_sizes.append(X.shape[0])
    hv_values = []
    avg_per_dim = []
    for sz in sample_sizes:
        hv = compute_hv_coverage(X[:sz])
        hv_values.append(hv)
        avg_per_dim.append(hv ** (1.0 / X.shape[1]))
        print(f"Sample size: {sz}, HV: {hv:.4f}, Per-dim: {avg_per_dim[-1]:.4f}")

    # plot both curves in different figures
    plt.figure()
    plt.plot(sample_sizes, hv_values, marker='o')
    plt.title('Hypervolume Coverage vs Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Hypervolume Coverage in dominated space')
    plt.grid()
    plt.savefig(os.path.join(result_dir , 'hv_coverage_vs_sample_size.png'))
    plt.close() 

    plt.figure()
    plt.plot(sample_sizes, avg_per_dim, marker='o', color='orange')
    plt.title('Average Per-Dimension Coverage vs Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Average Per-Dimension Coverage in dominated space')
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'avg_per_dimension_coverage_vs_sample_size.png'))
    plt.close()

if __name__ == "__main__":
    calculate_input_hv_coverage("/Users/id/Library/CloudStorage/OneDrive-BMWGroup/OpenSBT-ISU-Integration/isu/results/gemini-2.5_2000n_1i_03_00_00t_RS_configs_isu_features/rs")