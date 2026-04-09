import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.loader import load_data
from pipeline.tokenizer import tokenize
from smoothing.kneser_ney import get_prob, vocab

def tokenize_and_unk(text):
    tokens = tokenize(text)
    return [t if t in vocab else "<UNK>" for t in tokens]

def compute_perplexity(texts, get_prob_fn, order=3):
    total_log_prob = 0.0
    total_tokens   = 0
    epsilon        = 1e-10

    for text in texts:
        tokens = tokenize_and_unk(text)
        if len(tokens) < order:
            continue
        for i in range(order - 1, len(tokens)):
            word    = tokens[i]
            context = tuple(tokens[i - order + 1 : i])
            p = get_prob_fn(word, context, order)
            total_log_prob += math.log(max(p, epsilon))
            total_tokens   += 1

    if total_tokens == 0:
        return float("inf")

    return math.exp(-total_log_prob / total_tokens)

def get_prob_interpolated(word, context, order=3):
    l1, l2, l3 = 0.1, 0.3, 0.6
    return (
        l1 * get_prob(word, context, 1) +
        l2 * get_prob(word, context, 2) +
        l3 * get_prob(word, context, 3)
    )

if __name__ == "__main__":
    print("Loading data...")
    train, val, test = load_data()
    train     = train[:3000]
    val_texts = [t for t in val[:500] if t.strip()]

    pp_uni    = compute_perplexity(val_texts, get_prob, order=1)
    pp_bi     = compute_perplexity(val_texts, get_prob, order=2)
    pp_tri    = compute_perplexity(val_texts, get_prob, order=3)
    pp_interp = compute_perplexity(val_texts, get_prob_interpolated, order=3)

    print(f"\nResults (lower = better):")
    print(f"  Unigram      : {pp_uni:.2f}")
    print(f"  Bigram       : {pp_bi:.2f}")
    print(f"  Trigram      : {pp_tri:.2f}")
    print(f"  Interpolated : {pp_interp:.2f}")