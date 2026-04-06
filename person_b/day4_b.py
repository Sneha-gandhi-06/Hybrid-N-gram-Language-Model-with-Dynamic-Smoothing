import json
import sys
import os
import math
from collections import defaultdict, Counter

# ── Add root to path ─────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_a.day1.src.data_loader import load_data
from person_a.day2.src.counts import build_unigram, build_bigram, build_trigram
from person_b.day3_b import get_prob, vocab

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
train, val, test = load_data()
train = train[:3000]

# ── Tokenizer (same one Person A uses) ───────────────────────
from person_a.day1.src.tokenizer import tokenize

def tokenize_and_unk(text):
    tokens = tokenize(text)
    return [t if t in vocab else "<UNK>" for t in tokens]

# ── Perplexity ───────────────────────────────────────────────
def compute_perplexity(texts, get_prob_fn, order=3):
    total_log_prob = 0.0
    total_tokens   = 0
    epsilon        = 1e-10  # avoid log(0)

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

# ── Run evaluation ───────────────────────────────────────────
print("Running perplexity on validation set (first 500 lines)...")
val_texts = [t for t in val[:500] if t.strip()]

pp_tri = compute_perplexity(val_texts, get_prob, order=3)
pp_bi  = compute_perplexity(val_texts, get_prob, order=2)
pp_uni = compute_perplexity(val_texts, get_prob, order=1)

print(f"\nResults (lower = better):")
print(f"  Unigram  perplexity : {pp_uni:.2f}")
print(f"  Bigram   perplexity : {pp_bi:.2f}")
print(f"  Trigram  perplexity : {pp_tri:.2f}")

# ── Bonus: linear interpolation ──────────────────────────────
def get_prob_interpolated(word, context, order=3):
    l1, l2, l3 = 0.1, 0.3, 0.6   # weights, must sum to 1
    return (
        l1 * get_prob(word, context, 1) +
        l2 * get_prob(word, context, 2) +
        l3 * get_prob(word, context, 3)
    )

pp_interp = compute_perplexity(val_texts, get_prob_interpolated, order=3)
print(f"  Interpolated perplexity: {pp_interp:.2f}")

print("\nDay 4 complete! Commit and push:")
print("  git add person_b/day4_b.py")
print('  git commit -m "Day 4 done - perplexity evaluation implemented"')
print("  git push")