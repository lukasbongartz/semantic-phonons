import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import pca

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# The three scales shown in the main figure
MAIN_SCALES = ["storage_full", "time", "money"]


def _chirp_r2(coords, u):
    """Best single-PC mode-1 R² — matches the chirp_panel metric in the notebook."""
    sin1 = np.sin(np.pi / 2 * u)
    cos1 = np.cos(np.pi / 2 * u)
    X3   = np.column_stack([sin1, cos1, np.ones(len(u))])
    best = -np.inf
    for i in range(coords.shape[1]):
        pc = coords[:, i]
        w, _, _, _ = np.linalg.lstsq(X3, pc, rcond=None)
        fitted = X3 @ w
        r2 = 1.0 - np.sum((pc - fitted)**2) / np.sum((pc - pc.mean())**2)
        if r2 > best:
            best = r2
    return best


def permutation_test(words, log10_vals, embeddings, n_perms, rng):
    log10_vals = np.array(log10_vals)
    u = (log10_vals - log10_vals.min()) / (log10_vals.max() - log10_vals.min())

    k = min(len(words) - 1, 6)
    coords, _, _ = pca(np.array(embeddings), n_components=k)

    r2_obs  = _chirp_r2(coords, u)
    null_r2 = [_chirp_r2(coords[rng.permutation(len(words))], u)
               for _ in range(n_perms)]

    null_r2 = np.array(null_r2)
    p_value = float(np.mean(null_r2 >= r2_obs))
    return r2_obs, null_r2, p_value


def main():
    parser = argparse.ArgumentParser(description="Permutation test for log scale chirp")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    log_scale_path = RESULTS_DIR / "log_scale.json"
    if not log_scale_path.exists():
        print(f"log_scale.json not found at {log_scale_path}")
        print("Run experiments/log_scale.py first.")
        sys.exit(1)

    with open(log_scale_path) as f:
        data = json.load(f)
    scale_results = data["glove"]["scale_results"]

    rng = np.random.default_rng(args.seed)
    print(f"\nPermutation test — log scales  (n={args.n_permutations}, seed={args.seed})")
    print(f"{'Scale':<20}  {'obs R²':>7}  {'null mean':>9}  {'p-value':>8}")
    print("-" * 52)

    perm_results = {}
    for scale in MAIN_SCALES:
        sr = scale_results.get(scale)
        if sr is None or "embeddings" not in sr:
            print(f"{scale:<20}  (no data)")
            continue

        r2_obs, null_r2, p = permutation_test(
            sr["words"],
            sr["log10_values"],
            sr["embeddings"],
            args.n_permutations,
            rng,
        )
        print(f"{scale:<20}  {r2_obs:>7.3f}  {null_r2.mean():>9.3f}  {p:>8.4f}")
        perm_results[scale] = {
            "r2_obs":    r2_obs,
            "null_mean": float(null_r2.mean()),
            "null_std":  float(null_r2.std()),
            "p_value":   p,
        }

    out_path = RESULTS_DIR / "permutation_log_scale.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_permutations": args.n_permutations,
            "seed":           args.seed,
            "metric":         "mode-1 single-PC 3-param: a*sin(pi*u/2) + c*cos(pi*u/2) + b",
            "results":        perm_results,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()