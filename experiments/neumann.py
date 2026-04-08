import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.concepts import ORDINAL_SCALES, LLM_KEY_NEUMANN
from src.embeddings import load_glove, get_vectors, extract_llm_activations

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ── GloVe analysis ────────────────────────────────────────────────────────────

def analyze_scale(scale_name, word_ranks, W, word_to_idx):
    words_all = [w for w, _ in word_ranks]
    ranks_all = [r for _, r in word_ranks]

    found_words, vecs = get_vectors(words_all, W, word_to_idx)
    rank_map = dict(zip(words_all, ranks_all))
    ranks = np.array([rank_map[w] for w in found_words], dtype=float)

    N = len(found_words)
    if N < 5:
        return {"scale": scale_name, "n_found": N, "skipped": "too few words"}

    print(f"  {scale_name}: {N} words")

    return {
        "scale":         scale_name,
        "n_found":       N,
        "words":         found_words,
        "ordinal_ranks": ranks.tolist(),
        "embeddings":    vecs.tolist(),
    }


def run_glove():
    print("\n── GloVe 300d ──────────────────────────────────────────────────────")
    W, _, word_to_idx = load_glove()
    scale_results = {}
    for name, word_ranks in ORDINAL_SCALES.items():
        try:
            scale_results[name] = analyze_scale(name, word_ranks, W, word_to_idx)
        except Exception as e:
            scale_results[name] = {"scale": name, "error": str(e)}

    return {"scale_results": scale_results}


# ── LLM analysis ─────────────────────────────────────────────────────────────

TEMPLATES = {
    "quality":         "The level of quality is {word}",
    "certainty":       "The degree of certainty is {word}",
    "emotion_valence": "The emotional state is {word}",
    "temperature":     "The temperature feels {word}",
}


def run_llm(model_name, device):
    print(f"\n── {model_name} ────────────────────────────────────────────────────")
    N_LAYERS = 19

    all_acts = {}
    all_meta = {}

    for llm_key, scale_key in LLM_KEY_NEUMANN.items():
        word_ranks = ORDINAL_SCALES[scale_key]
        words = [w for w, _ in word_ranks]
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
        all_meta[llm_key] = {"words": words}

    if all_acts:
        npz_path = RESULTS_DIR / "llm_neumann.npz"
        np.savez(str(npz_path), activations=all_acts, meta=all_meta)
        print(f"\nSaved activations → {npz_path}")

    return {"scales_extracted": list(all_acts.keys())}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neumann BC / ordinal scales")
    parser.add_argument("--llm",    action="store_true", help="Also run Gemma 2B")
    parser.add_argument("--model",  default="google/gemma-2b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = {
        "experiment": "neumann_ordinal",
        "theory":     "Neumann BC → cosine modes → parabola in top-2 PC plane",
        "glove":      run_glove(),
    }

    if args.llm:
        output["llm"] = run_llm(args.model, args.device)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "neumann.json"
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