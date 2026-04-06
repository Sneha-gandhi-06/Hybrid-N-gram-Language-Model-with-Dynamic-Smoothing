import json
import sys
import os
import math
from collections import defaultdict, Counter

# ── Add root to path ─────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_a.day1.src.data_loader import load_data
from person_a.day1.src.tokenizer import tokenize
from person_a.day2.src.counts import build_unigram, build_bigram, build_trigram
from person_b.day3_b import get_prob, vocab, bi_counts, tri_counts, uni_counts
from person_b.day4_b import compute_perplexity

# ── Rebuild counts ───────────────────────────────────────────
print("Loading data & building counts...")
train, val, test = load_data()
train = train[:3000]

# ── Tokenizer ────────────────────────────────────────────────
def tokenize_and_unk(text):
    tokens = tokenize(text)
    return [t if t in vocab else "<UNK>" for t in tokens]

# ── get_count() for Person A's switcher ──────────────────────
# Person A calls this to check context confidence
def get_count(context: tuple) -> int:
    """
    Returns how many times this context was seen in training.
    context = ()        → total unigram tokens
    context = (w,)      → bigram context count
    context = (w1, w2)  → trigram context count
    """
    if len(context) == 0:
        return sum(uni_counts.values())
    elif len(context) == 1:
        return sum(bi_counts.get(context, {}).values())
    else:
        return sum(tri_counts.get(context[-2:], {}).values())

# ── Dynamic switcher (mirrors Person A's logic) ──────────────
T1 = 10   # trigram confidence threshold
T2 = 5    # bigram confidence threshold

def hybrid_get_prob(word: str, context: tuple, order=3) -> float:
    """
    Dynamically picks N-gram order based on context confidence.
    This is what makes the model 'hybrid'.
    """
    tri_ctx = tuple(context[-2:]) if len(context) >= 2 else ()
    bi_ctx  = tuple(context[-1:]) if len(context) >= 1 else ()

    if get_count(tri_ctx) >= T1:
        return get_prob(word, context, 3)
    elif get_count(bi_ctx) >= T2:
        return get_prob(word, context, 2)
    else:
        return get_prob(word, context, 1)

# ── Run full comparison ──────────────────────────────────────
print("Running perplexity comparison on validation set...")
val_texts = [t for t in val[:500] if t.strip()]

pp_static_trigram = compute_perplexity(val_texts, get_prob, order=3)
pp_hybrid         = compute_perplexity(val_texts, hybrid_get_prob, order=3)

print(f"\nResults (lower = better):")
print(f"  Static KN trigram : {pp_static_trigram:.2f}")
print(f"  Hybrid dynamic    : {pp_hybrid:.2f}")

improvement = ((pp_static_trigram - pp_hybrid) / pp_static_trigram) * 100
print(f"  Improvement       : {improvement:.1f}%")

# ── End to end next-word prediction demo ─────────────────────
print("\n--- Next Word Prediction Demo ---")

def predict_next(sentence, top_n=5):
    tokens  = tokenize_and_unk(sentence)
    context = tuple(tokens[-2:])
    scores  = {}
    for word in list(vocab.keys())[:5000]:   # check top 5000 vocab words
        scores[word] = hybrid_get_prob(word, context)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nInput   : '{sentence}'")
    print(f"Context : {context}")
    print(f"Top {top_n} predictions:")
    for word, prob in top:
        print(f"  {word:<20} {prob:.6f}")

predict_next("the cat sat on the")
predict_next("in the united")
predict_next("he was looking at")

print("\nDay 5 complete! Commit and push:")
print("  git add person_b/day5_b.py")
print('  git commit -m "Day 5 done - integration and hybrid model working"')
print("  git push")