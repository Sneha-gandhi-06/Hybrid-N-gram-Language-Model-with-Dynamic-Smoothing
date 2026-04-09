import json
import sys
import os
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.loader import load_data
from pipeline.counts import build_unigram, build_bigram, build_trigram

# ── Load vocab ───────────────────────────────────────────────
with open("data/vocab.json") as f:
    vocab = json.load(f)

V = len(vocab)

# ── Build counts ─────────────────────────────────────────────
train, _, _ = load_data()
train = train[:20000]

uni_counts = build_unigram(train)
bi_raw     = build_bigram(train)
tri_raw    = build_trigram(train)

bi_counts = defaultdict(Counter)
for (w1, w2), count in bi_raw.items():
    bi_counts[(w1,)][w2] += count

tri_counts = defaultdict(Counter)
for (w1, w2, w3), count in tri_raw.items():
    tri_counts[(w1, w2)][w3] += count

# ── Continuation counts ──────────────────────────────────────
continuation_counts = Counter()
for context, word_counts in bi_counts.items():
    for word in word_counts:
        continuation_counts[word] += 1

total_continuation = sum(continuation_counts.values())

D = 0.75

# ── KN functions ─────────────────────────────────────────────
def kn_unigram(word):
    return continuation_counts.get(word, 0) / total_continuation

def kn_bigram(word, context):
    ctx       = context[-1:]
    ctx_total = sum(bi_counts.get(ctx, {}).values())
    if ctx_total == 0:
        return kn_unigram(word)
    word_count  = bi_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(bi_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total
    return discounted + lambda_ctx * kn_unigram(word)

def kn_trigram(word, context):
    ctx       = context[-2:]
    ctx_total = sum(tri_counts.get(ctx, {}).values())
    if ctx_total == 0:
        return kn_bigram(word, context[-1:])
    word_count  = tri_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(tri_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total
    return discounted + lambda_ctx * kn_bigram(word, context[-1:])

def get_prob(word: str, context: tuple, order: int) -> float:
    word    = word if word in vocab else "<UNK>"
    context = tuple(w if w in vocab else "<UNK>" for w in context)
    if order == 1:
        return kn_unigram(word)
    elif order == 2:
        return kn_bigram(word, context[-1:])
    else:
        return kn_trigram(word, context[-2:])

if __name__ == "__main__":
    sample_vocab = list(vocab.keys())[:2000]
    total = sum(get_prob(w, ("on", "the"), 3) for w in sample_vocab)
    print(f"Prob sum over 2000 words: {total:.4f}  ← should be ~1.0")
    for w in ["the", "a", "mat", "floor", "<UNK>"]:
        print(f"  P({w} | on, the) = {get_prob(w, ('on','the'), 3):.8f}")