import json
import pickle
import math
from collections import defaultdict, Counter

# ── Load files ───────────────────────────────────────────────
with open("data/vocab.json") as f:
    vocab = json.load(f)

with open("data/counts.pkl", "rb") as f:
    data = pickle.load(f)
    uni_counts = data["unigram"]
    bi_counts  = data["bigram"]
    tri_counts = data["trigram"]

V = len(vocab)
print(f"Vocab size: {V}")

# ── Step 1: Continuation counts ──────────────────────────────
# For each word, count how many unique LEFT contexts it appears in
# e.g. "francisco" only follows "san" → low continuation
# e.g. "dog" follows many words → high continuation
continuation_counts = Counter()
for context, word_counts in bi_counts.items():
    for word in word_counts:
        continuation_counts[word] += 1

total_continuation = sum(continuation_counts.values())
print(f"Continuation counts built: {len(continuation_counts)} words")

# ── Step 2: KN unigram ───────────────────────────────────────
def kn_unigram(word):
    return continuation_counts.get(word, 0) / total_continuation

# ── Step 3: KN bigram ────────────────────────────────────────
D = 0.75  # discount — standard value, works well in practice

def kn_bigram(word, context):
    ctx = context[-1:]
    ctx_total = sum(bi_counts.get(ctx, {}).values())

    if ctx_total == 0:
        return kn_unigram(word)

    word_count  = bi_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(bi_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total

    return discounted + lambda_ctx * kn_unigram(word)

# ── Step 4: KN trigram ───────────────────────────────────────
def kn_trigram(word, context):
    ctx = context[-2:]
    ctx_total = sum(tri_counts.get(ctx, {}).values())

    if ctx_total == 0:
        return kn_bigram(word, context[-1:])

    word_count  = tri_counts.get(ctx, {}).get(word, 0)
    discounted  = max(word_count - D, 0) / ctx_total
    n_following = len(tri_counts.get(ctx, {}))
    lambda_ctx  = (D * n_following) / ctx_total

    return discounted + lambda_ctx * kn_bigram(word, context[-1:])

# ── Unified get_prob() ───────────────────────────────────────
def get_prob(word: str, context: tuple, order: int) -> float:
    word    = word if word in vocab else "<UNK>"
    context = tuple(w if w in vocab else "<UNK>" for w in context)
    if order == 1:
        return kn_unigram(word)
    elif order == 2:
        return kn_bigram(word, context[-1:])
    else:
        return kn_trigram(word, context[-2:])

# ── Sanity checks ────────────────────────────────────────────
print("\n--- Sanity Checks ---")

# Check 1: probs sum to ~1
sample_vocab = list(vocab.keys())[:2000]
total = sum(get_prob(w, ("on", "the"), 3) for w in sample_vocab)
print(f"Prob sum over 2000 words (trigram): {total:.4f}  ← should be close to 1.0")

# Check 2: compare KN vs raw frequency
print("\nKN trigram probs for context ('on', 'the'):")
for w in ["mat", "floor", "cat", "the", "<UNK>"]:
    print(f"  P({w} | on, the) = {get_prob(w, ('on','the'), 3):.8f}")

# Check 3: all three orders
print("\nAll three orders for word='mat':")
print(f"  Unigram : {get_prob('mat', (), 1):.8f}")
print(f"  Bigram  : {get_prob('mat', ('the',), 2):.8f}")
print(f"  Trigram : {get_prob('mat', ('on','the'), 3):.8f}")

print("\nDay 3 complete!")