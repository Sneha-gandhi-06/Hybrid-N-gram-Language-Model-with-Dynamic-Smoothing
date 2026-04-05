import json
import pickle
import math
from collections import defaultdict, Counter

# ── Load Person A's files ────────────────────────────────────
with open("../person_a/vocab.json") as f:
    vocab = json.load(f)

with open("../person_a/counts.pkl", "rb") as f:
    data = pickle.load(f)
    uni_counts = data["unigram"]   # Counter
    bi_counts  = data["bigram"]    # defaultdict(Counter)
    tri_counts = data["trigram"]   # defaultdict(Counter)

V = len(vocab)
print(f"Vocab size: {V}")
print(f"Unigram entries: {len(uni_counts)}")
print(f"Bigram contexts: {len(bi_counts)}")
print(f"Trigram contexts: {len(tri_counts)}")

# ── Add-k Smoothing ──────────────────────────────────────────
# Formula: P(w|ctx) = (count(ctx,w) + k) / (count(ctx) + k*V)
# k=1 is called Laplace smoothing — try other values later

k = 1.0

def addk_unigram(word):
    total = sum(uni_counts.values())
    return (uni_counts.get(word, 0) + k) / (total + k * V)

def addk_bigram(word, context):
    # context = (prev_word,)
    ctx = context[-1:]
    ctx_total = sum(bi_counts.get(ctx, {}).values())
    word_count = bi_counts.get(ctx, {}).get(word, 0)
    return (word_count + k) / (ctx_total + k * V)

def addk_trigram(word, context):
    # context = (prev_prev_word, prev_word)
    ctx = context[-2:]
    ctx_total = sum(tri_counts.get(ctx, {}).values())
    word_count = tri_counts.get(ctx, {}).get(word, 0)
    return (word_count + k) / (ctx_total + k * V)

# ── Unified get_prob() interface ─────────────────────────────
def get_prob(word: str, context: tuple, order: int) -> float:
    word    = word if word in vocab else "<UNK>"
    context = tuple(w if w in vocab else "<UNK>" for w in context)
    if order == 1:
        return addk_unigram(word)
    elif order == 2:
        return addk_bigram(word, context[-1:])
    else:
        return addk_trigram(word, context[-2:])

# ── Sanity checks ────────────────────────────────────────────
print("\n--- Sanity Checks ---")

# Check 1: probs must sum to ~1.0 over full vocab
# (expensive on full vocab so we sample 2000 words)
sample_vocab = list(vocab.keys())[:2000]
ctx = ("on", "the")
total = sum(get_prob(w, ctx, 3) for w in sample_vocab)
print(f"Prob sum over 2000 words (trigram): {total:.4f}  ← should be close to 1.0")

# Check 2: common words should get higher probs than rare ones
print("\nSample trigram probs for context ('on', 'the'):")
test_words = ["mat", "floor", "cat", "the", "xqzjw", "<UNK>"]
for w in test_words:
    print(f"  P({w} | on, the) = {get_prob(w, ('on','the'), 3):.8f}")

# Check 3: all three orders work
print("\nAll three orders for word='mat':")
print(f"  Unigram:  {get_prob('mat', (), 1):.8f}")
print(f"  Bigram:   {get_prob('mat', ('the',), 2):.8f}")
print(f"  Trigram:  {get_prob('mat', ('on','the'), 3):.8f}")

print("\nDay 2 complete! Commit and push:")