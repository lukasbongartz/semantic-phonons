import numpy as np

_glove_cache = {}


def load_glove(vocab_size=400_000, dim=300):
    """Load GloVe via gensim's model downloader. Downloads on first call."""
    key = (vocab_size, dim)
    if key in _glove_cache:
        return _glove_cache[key]

    import gensim.downloader as api
    model_name = f"glove-wiki-gigaword-{dim}"
    print(f"Loading {model_name} ...")
    glove = api.load(model_name)
    n = min(vocab_size, len(glove.index_to_key))
    vocab = list(glove.index_to_key[:n])
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    W = np.array([glove[w] for w in vocab], dtype=np.float32)
    print(f"  {n} words, dim={dim}")
    _glove_cache[key] = (W, vocab, word_to_idx)
    return W, vocab, word_to_idx


def get_vectors(words, W, word_to_idx):
    """Return (found_words, matrix) for the subset of words in the vocabulary."""
    found, vecs = [], []
    for w in words:
        if w in word_to_idx:
            found.append(w)
            vecs.append(W[word_to_idx[w]])
        else:
            print(f"  [not in vocab] {w}")
    if not found:
        raise ValueError("None of the requested words are in the vocabulary.")
    return found, np.array(vecs, dtype=np.float64)


def extract_llm_activations(prompts, model_name="google/gemma-2b",
                             layers=None, device="cuda", batch_size=8):
    """
    Extract hidden states from a causal LM using last-token pooling
    (Karkada et al.): for each prompt, take the hidden state at the
    final non-padding position.

    Returns: {layer_idx: ndarray of shape (N, hidden_dim)}
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_bf16 = "27b" in model_name.lower() or "9b" in model_name.lower()
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Loading {model_name} (dtype={load_dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=load_dtype,
        device_map=device,
    )
    model.config.output_hidden_states = True
    model.eval()
    n_layers = model.config.num_hidden_layers

    if layers is None:
        layers = list(range(n_layers + 1))
    layers = [l if l >= 0 else n_layers + 1 + l for l in layers]

    result = {l: [] for l in layers}

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**enc)

        last_idx = enc["attention_mask"].sum(dim=1).long() - 1

        for l in layers:
            h = out.hidden_states[l]
            for b in range(len(batch_prompts)):
                act = h[b, last_idx[b], :].float().cpu().numpy()
                result[l].append(act)

        print(f"  {min(start + batch_size, len(prompts))}/{len(prompts)} prompts done")

    return {l: np.array(v) for l, v in result.items()}
