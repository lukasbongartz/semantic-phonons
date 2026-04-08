import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.concepts import LOG_SCALES, LLM_KEY_LOG
from src.embeddings import load_glove, get_vectors, extract_llm_activations

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ── GloVe analysis ────────────────────────────────────────────────────────────

def analyze_scale(scale_name, word_values, W, word_to_idx):
    words_all = [w for w, _ in word_values]
    vals_all  = [v for _, v in word_values]

    found_words, vecs = get_vectors(words_all, W, word_to_idx)
    val_map = dict(zip(words_all, vals_all))
    vals = np.array([val_map[w] for w in found_words], dtype=float)

    N = len(found_words)
    if N < 4:
        return {"scale": scale_name, "n_found": N, "skipped": "too few words"}

    print(f"  {scale_name}: {N} words")
    log10_vals = np.log10(vals)

    return {
        "scale":          scale_name,
        "n_found":        N,
        "words":          found_words,
        "log10_values":   log10_vals.tolist(),
        "embeddings":     vecs.tolist(),
    }


def run_glove():
    print("\n── GloVe 300d ──────────────────────────────────────────────────────")
    W, _, word_to_idx = load_glove()
    scale_results = {}
    for name, word_vals in LOG_SCALES.items():
        try:
            scale_results[name] = analyze_scale(name, word_vals, W, word_to_idx)
        except Exception as e:
            scale_results[name] = {"scale": name, "error": str(e)}

    return {"scale_results": scale_results}


# ── LLM analysis ─────────────────────────────────────────────────────────────

TEMPLATES = {
    "storage_full": "The amount of storage is one {word}",
    "time":         "The duration of time is one {word}",
    "money":        "The sum of money is one {word}",
}


def run_llm(model_name, device):
    print(f"\n── {model_name} ────────────────────────────────────────────────────")
    N_LAYERS = 19

    all_acts = {}
    all_meta = {}

    for llm_key, scale_key in LLM_KEY_LOG.items():
        word_vals = LOG_SCALES[scale_key]
        words  = [w for w, _ in word_vals]
        values = [v for _, v in word_vals]
        template = TEMPLATES[scale_key]
        prompts = [template.format(word=w) for w in words]

        print(f"\n  Extracting '{scale_key}' ({len(words)} words) ...")
        try:
            acts = extract_llm_activations(
                prompts, model_name=model_name,
                layers=list(range(N_LAYERS)), device=device,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

        all_acts[llm_key] = {str(l): v for l, v in acts.items()}
        all_meta[llm_key] = {"words": words, "values": values}

    if all_acts:
        npz_path = RESULTS_DIR / "llm_log_scale.npz"
        np.savez(str(npz_path), activations=all_acts, meta=all_meta)
        print(f"\nSaved activations → {npz_path}")

    return {"scales_extracted": list(all_acts.keys())}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Logarithmic scales / chirp geometry")
    parser.add_argument("--llm",    action="store_true", help="Also run Gemma 2B")
    parser.add_argument("--model",  default="google/gemma-2b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = {
        "experiment": "log_scale_chirp",
        "theory":     "log-uniform spacing → chirp eigenmodes → sin(πu/2) in log coords",
        "glove":      run_glove(),
    }

    if args.llm:
        output["llm"] = run_llm(args.model, args.device)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "log_scale.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nSaved → {path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


if __name__ == "__main__":
    main()