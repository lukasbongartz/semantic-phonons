import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.concepts import EMOTION_CIRCUMPLEX, LLM_KEY_NEUMANN_2D
from src.embeddings import load_glove, get_vectors, extract_llm_activations

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

TEMPLATE = "The person is feeling {word}"


# ── GloVe analysis ────────────────────────────────────────────────────────────

def run_glove():
    print("\n── GloVe 300d ──────────────────────────────────────────────────────")
    W, _, word_to_idx = load_glove()

    words_raw = [w for w, _, _ in EMOTION_CIRCUMPLEX]
    valence   = [v for _, v, _ in EMOTION_CIRCUMPLEX]
    arousal   = [a for _, _, a in EMOTION_CIRCUMPLEX]

    found_words, vecs = get_vectors(words_raw, W, word_to_idx)

    found_set = set(found_words)
    keep = [i for i, w in enumerate(words_raw) if w in found_set]
    valence_found = [valence[i] for i in keep]
    arousal_found = [arousal[i] for i in keep]

    print(f"  emotion_circumplex: {len(found_words)} words")

    return {
        "scale_results": {
            "emotion_circumplex": {
                "scale":      "emotion_circumplex",
                "n_found":    len(found_words),
                "words":      found_words,
                "valence":    valence_found,
                "arousal":    arousal_found,
                "embeddings": vecs.tolist(),
            }
        }
    }


# ── LLM analysis ─────────────────────────────────────────────────────────────

def run_llm(model_name, device, batch_size):
    print(f"\n── {model_name} ────────────────────────────────────────────────────")

    words_raw = [w for w, _, _ in EMOTION_CIRCUMPLEX]
    valence   = [v for _, v, _ in EMOTION_CIRCUMPLEX]
    arousal   = [a for _, _, a in EMOTION_CIRCUMPLEX]

    prompts = [TEMPLATE.format(word=w) for w in words_raw]

    print(f"\n  Extracting emotion_circumplex ({len(words_raw)} words) ...")
    try:
        acts = extract_llm_activations(
            prompts, model_name=model_name, device=device,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}

    llm_key = LLM_KEY_NEUMANN_2D
    all_acts = {llm_key: {str(l): v for l, v in acts.items()}}
    all_meta = {llm_key: {
        "words":   words_raw,
        "valence": valence,
        "arousal": arousal,
    }}

    npz_path = RESULTS_DIR / "llm_neumann_2d.npz"
    np.savez(str(npz_path), activations=all_acts, meta=all_meta)
    print(f"\nSaved activations → {npz_path}")

    return {"scales_extracted": [llm_key]}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2-D Neumann BC / Russell emotion circumplex")
    parser.add_argument("--llm",        action="store_true",
                        help="Also run LLM extraction")
    parser.add_argument("--model",      default="google/gemma-2-27b")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Reduce for large models to avoid OOM")
    args = parser.parse_args()

    output = {
        "experiment": "neumann_2d_circumplex",
        "theory":     "2-D Neumann BC → eigenmodes cos(mπu)cos(nπv) on valence×arousal square",
        "glove":      run_glove(),
    }

    if args.llm:
        output["llm"] = run_llm(args.model, args.device, args.batch_size)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "neumann_2d.json"
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
